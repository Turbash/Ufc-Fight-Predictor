import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_models():
    print("Loading data...")
    # Load the dataset
    df = pd.read_csv("data/ufc-master.csv")
    
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
    
    # Using XGBoost for winner prediction
    xgb_winner = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=2,
        eval_metric='mlogloss',
        random_state=42,
        use_label_encoder=False
    )
    
    # Train the model
    xgb_winner.fit(X_train_winner, y_train_winner)
    
    # Evaluate the model
    y_pred_winner = xgb_winner.predict(X_test_winner)
    accuracy = accuracy_score(y_test_winner, y_pred_winner)
    print(f"Winner Model Accuracy: {accuracy:.4f}")
    print(classification_report(y_test_winner, y_pred_winner))
    
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
    
    # Split data
    X_train_finish, X_test_finish, y_train_finish, y_test_finish = train_test_split(
        X_finish, y_finish, test_size=0.2, random_state=42
    )
    
    # Scale the data for logistic regression
    scaler = StandardScaler()
    X_train_finish_scaled = scaler.fit_transform(X_train_finish)
    X_test_finish_scaled = scaler.transform(X_test_finish)
    
    # Train logistic regression model
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_finish_scaled, y_train_finish)
    
    # Evaluate the model
    y_pred_finish = log_reg.predict(X_test_finish_scaled)
    accuracy = accuracy_score(y_test_finish, y_pred_finish)
    print(f"Finish Model Accuracy: {accuracy:.4f}")
    print(classification_report(y_test_finish, y_pred_finish))
    
    # Save models
    print("Saving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_winner, 'models/winner_model.pkl')
    joblib.dump(log_reg, 'models/finish_model.pkl')
    joblib.dump(scaler, 'models/finish_scaler.pkl')
    
    # Also save feature lists to avoid mismatch in future
    joblib.dump(list(X_winner.columns), 'models/winner_features.pkl')
    joblib.dump(list(X_finish.columns), 'models/finish_features.pkl')
    print("Models trained and saved successfully!")

if __name__ == "__main__":
    train_models()
