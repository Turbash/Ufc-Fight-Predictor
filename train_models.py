import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import joblib
import os

def train_models():
    print("Loading data...")
    # Load the dataset
    df = pd.read_csv("assets/ufc-master.csv")
    
    # Create models directory at the beginning to avoid FileNotFoundError
    print("Setting up directories...")
    os.makedirs('models', exist_ok=True)
    
    print("Preprocessing data...")
    # Basic preprocessing (handle missing values)
    # Fill rank columns with 0 (not ranked)
    rank_columns = [col for col in df.columns if 'weight' in col.lower() and 'rank' in col.lower()]
    for col in rank_columns:
        df[col] = df[col].fillna(0)
    
    # Drop rows with missing odds
    df = df.dropna(subset=['RedOdds', 'BlueOdds'])
    
    # Fill other numerical columns with median
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col not in rank_columns]
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns
    df['BlueStance'] = df['BlueStance'].fillna('Orthodox')
    df['FinishRoundTime'] = df['FinishRoundTime'].fillna('0:00')
    df['Finish'] = df['Finish'].fillna('Unknown')
    df['FinishDetails'] = df['FinishDetails'].fillna('Unknown')
    
    # Drop non-informative columns
    columns_to_drop = [
        'RedFighter', 'BlueFighter',
        'Date', 'Location', 'Country', 'EmptyArena'
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Create WinnerNumeric column (Red = 0, Blue = 1)
    df['WinnerNumeric'] = df['Winner'].apply(lambda x: 0 if x == 'Red' else 1)
    
    # Create FinishNumeric column (0 = KO/TKO, 1 = SUB, 2 = DEC)
    finish_mapping = {
        'KO/TKO': 0,
        'SUB': 1,
        'U-DEC': 2,
        'S-DEC': 2,
        'M-DEC': 2,
        'DQ': 2,
        'Unknown': 2,
        'Overturned': 2
    }
    df['FinishNumeric'] = df['Finish'].map(finish_mapping)
    
    # Convert 'FinishRoundTime' from 'minutes:seconds' format to total seconds
    def convert_to_seconds(time_str):
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds
        except:
            return 0
            
    df['FinishRoundTimeSecs'] = df['FinishRoundTime'].apply(convert_to_seconds)
    
    # Remove post-fight information that would cause data leakage
    leakage_columns = [
        'FinishRound',
        'TotalFightTimeSecs',
        'FinishRoundTimeSecs',
        'Winner',
        'Finish',
        'FinishDetails'
    ]
    df = df.drop(columns=[col for col in leakage_columns if col in df.columns])
    
    print("Training winner prediction model...")
    # Winner prediction model (SVM)
    winner_features = [
        'RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue',
        'TitleBout', 'NumberOfRounds', 'RedCurrentLoseStreak', 'BlueCurrentLoseStreak',
        'RedCurrentWinStreak', 'BlueCurrentWinStreak', 'RedAvgSigStrLanded', 'BlueAvgSigStrLanded',
        'RedAvgSigStrPct', 'BlueAvgSigStrPct', 'RedAvgSubAtt', 'BlueAvgSubAtt',
        'RedAvgTDLanded', 'BlueAvgTDLanded', 'RedWinsByKO', 'BlueWinsByKO',
        'RedWinsBySubmission', 'BlueWinsBySubmission', 'RedWinsByDecisionUnanimous',
        'BlueWinsByDecisionUnanimous', 'RedWins', 'BlueWins', 'RedHeightCms',
        'BlueHeightCms', 'RedReachCms', 'BlueReachCms', 'RedWeightLbs', 'BlueWeightLbs',
        'WinStreakDif', 'LoseStreakDif', 'HeightDif', 'ReachDif', 'AgeDif', 'SigStrDif',
        'AvgSubAttDif', 'AvgTDDif'
    ]
    
    # Prepare dataset
    df_winner = df[[col for col in winner_features + ['WinnerNumeric'] if col in df.columns]]
    X_winner = df_winner.drop(columns=['WinnerNumeric'])
    y_winner = df_winner['WinnerNumeric']
    
    # Split data
    X_train_winner, X_test_winner, y_train_winner, y_test_winner = train_test_split(
        X_winner, y_winner, test_size=0.2, random_state=42
    )
    
    # Step 1: Ensure that only numerical columns are selected
    X_train_numeric = X_train_winner.select_dtypes(include=[np.number])
    
    # Save the exact columns used for training - this is critical for prediction consistency
    winner_numeric_columns = list(X_train_numeric.columns)

    # Step 2: Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = X_train_numeric.quantile(0.25)
    Q3 = X_train_numeric.quantile(0.75)
    IQR = Q3 - Q1

    # Step 3: Detect outliers
    outliers = ((X_train_numeric < (Q1 - 1.5 * IQR)) | (X_train_numeric > (Q3 + 1.5 * IQR)))

    # Step 4: Remove outliers from the data
    X_train_no_outliers = X_train_numeric[~outliers.any(axis=1)]
    y_train_no_outliers = y_train_winner[~outliers.any(axis=1)]

    # Step 5: Standardize the data (scaling)
    winner_scaler = StandardScaler()
    X_train_scaled = winner_scaler.fit_transform(X_train_no_outliers)

    # Step 6: Train the SVM model
    svm = SVC(kernel='linear', random_state=42, probability=True)
    svm.fit(X_train_scaled, y_train_no_outliers)

    # Step 7: Make predictions
    X_test_numeric = X_test_winner[winner_numeric_columns]  # Use the same columns as training
    y_pred_winner = svm.predict(winner_scaler.transform(X_test_numeric))

    # Step 8: Evaluate the model
    print("SVM Model (No Outliers) - Accuracy:", accuracy_score(y_test_winner, y_pred_winner))
    print("\nSVM Model (No Outliers) - Classification Report:\n", classification_report(y_test_winner, y_pred_winner))
    print("\nSVM Model (No Outliers) - Confusion Matrix:\n", confusion_matrix(y_test_winner, y_pred_winner))

    # Calculate and plot ROC curve
    y_scores = svm.decision_function(winner_scaler.transform(X_test_numeric))
    roc_auc = roc_auc_score(y_test_winner, y_scores)
    print(f"AUC-ROC Score: {roc_auc:.4f}")

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test_winner, y_scores)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Random guessing line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (SVM, No Outliers)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('models/winner_model_roc.png')
    plt.close()
    
    print("Training finish type prediction model...")
    # Finish prediction model (Logistic Regression)
    finish_features = [
        'RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue',
        'TitleBout', 'NumberOfRounds', 'RedCurrentLoseStreak', 'BlueCurrentLoseStreak',
        'RedCurrentWinStreak', 'BlueCurrentWinStreak', 'RedAvgSigStrLanded', 'BlueAvgSigStrLanded',
        'RedAvgSigStrPct', 'BlueAvgSigStrPct', 'RedAvgSubAtt', 'BlueAvgSubAtt',
        'RedAvgTDLanded', 'BlueAvgTDLanded', 'RedWinsByKO', 'BlueWinsByKO',
        'RedWinsBySubmission', 'BlueWinsBySubmission', 'RedWinsByDecisionUnanimous',
        'BlueWinsByDecisionUnanimous', 'RedWins', 'BlueWins', 'RedHeightCms',
        'BlueHeightCms', 'RedReachCms', 'BlueReachCms', 'RedWeightLbs', 'BlueWeightLbs',
        'WinStreakDif', 'LoseStreakDif', 'HeightDif', 'ReachDif', 'AgeDif', 'SigStrDif',
        'AvgSubAttDif', 'AvgTDDif', 'RedDecOdds', 'BlueDecOdds', 'RSubOdds', 'BSubOdds',
        'RKOOdds', 'BKOOdds'
    ]
    
    # Prepare dataset
    df_finish = df[[col for col in finish_features + ['FinishNumeric'] if col in df.columns]]
    X_finish = df_finish.drop(columns=['FinishNumeric'])
    y_finish = df_finish['FinishNumeric']
    
    # Save the exact columns used for finish prediction - for consistency
    finish_numeric_columns = list(X_finish.columns)
    
    # 1. Standardize the features
    finish_scaler = StandardScaler()
    X_scaled = finish_scaler.fit_transform(X_finish)

    # 2. Split the data
    X_train_finish, X_test_finish, y_train_finish, y_test_finish = train_test_split(
        X_scaled, y_finish, test_size=0.2, random_state=42
    )

    # 3. Fit Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_finish, y_train_finish)

    # 4. Predictions and Evaluation
    y_pred_finish = log_reg.predict(X_test_finish)
    accuracy = accuracy_score(y_test_finish, y_pred_finish)
    print(f"Logistic Regression Finish Prediction - Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test_finish, y_pred_finish))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test_finish, y_pred_finish))

    # 5. AUC-ROC Score (for multiclass)
    y_proba = log_reg.predict_proba(X_test_finish)  # Probabilities for all classes
    auc_score = roc_auc_score(y_test_finish, y_proba, multi_class='ovr')  # Multi-class One-vs-Rest strategy
    print(f"\nAUC-ROC Score (Multiclass OVR): {auc_score:.4f}")

    # 6. Plot AUC-ROC Curve
    # Binarize the labels for ROC curve
    classes = np.unique(y_test_finish)
    y_test_bin = label_binarize(y_test_finish, classes=classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'green', 'red']
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve for class {classes[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Multiclass ROC Curve (Finish Prediction)', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig('models/finish_model_roc.png')
    plt.close()
    
    # Save models
    print("Saving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(svm, 'models/winner_model.pkl')
    joblib.dump(winner_scaler, 'models/winner_scaler.pkl')
    joblib.dump(log_reg, 'models/finish_model.pkl')
    joblib.dump(finish_scaler, 'models/finish_scaler.pkl')
    
    # Save the exact feature lists to ensure prediction uses the same features
    joblib.dump(winner_numeric_columns, 'models/winner_features.pkl')
    joblib.dump(finish_numeric_columns, 'models/finish_features.pkl')
    print("Models trained and saved successfully!")

if __name__ == "__main__":
    train_models()
