import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                          roc_auc_score, roc_curve, brier_score_loss, log_loss)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
import joblib
import os
import sys
from datetime import datetime
# Try to import SHAP - used for model explanations
try:
    import shap
except ImportError:
    print("Warning: SHAP library not found. Install with: pip install shap")
    print("Model explanations will be limited without SHAP.")

def get_paired_features(feature_names):
    """
    Create a mapping between blue and red feature pairs.
    Assumes features follow patterns like 'Blue...' and 'Red...'
    """
    if isinstance(feature_names, pd.Index):
        feature_names = feature_names.tolist()
        
    blue_features = [f for f in feature_names if f.startswith('Blue')]
    red_features = [f for f in feature_names if f.startswith('Red')]
    
    pairs = {}
    for blue_feat in blue_features:
        feat_name = blue_feat[4:]
        red_feat = f'Red{feat_name}'
        if red_feat in red_features:
            pairs[blue_feat] = red_feat
            pairs[red_feat] = blue_feat
    
    return pairs

def paired_feature_selection(X, y, feature_names, k=20):
    """
    Perform feature selection ensuring that if a feature is selected,
    its counterpart is also selected.
    """
    if isinstance(feature_names, pd.Index):
        feature_names_list = feature_names.tolist()
    else:
        feature_names_list = list(feature_names)
    
    pairs = get_paired_features(feature_names_list)
    
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    
    scores = selector.scores_
    feature_scores = list(zip(feature_names_list, scores))
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    selected = set()
    for feature, score in feature_scores:
        if len(selected) >= k:
            break
        
        if feature not in selected:
            selected.add(feature)
            if feature in pairs:
                paired_feature = pairs[feature]
                selected.add(paired_feature)
    
    selected_features = sorted(list(selected), 
                              key=lambda x: feature_names_list.index(x))
    
    return selected_features[:k]

def engineer_features(df):
    """
    Add engineered features to the dataframe with enhanced finish prediction capabilities
    """
    print("\nPerforming advanced feature engineering...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df_eng = df.copy()
    
    # 1. Log-odds transformation for betting odds
    print("Creating log-odds transformation...")
    
    if all(col in df_eng.columns for col in ['RedOdds', 'BlueOdds']):
        # Calculate log-odds ratio
        df_eng['LogOddsRatio'] = np.log(np.abs(df_eng['RedOdds']) / np.abs(df_eng['BlueOdds']))
        
        # Handle potential infinities or NaN from division
        df_eng['LogOddsRatio'] = df_eng['LogOddsRatio'].replace([np.inf, -np.inf], np.nan)
        df_eng['LogOddsRatio'] = df_eng['LogOddsRatio'].fillna(0)
    
    # 2. Create derivative features that emphasize the difference or ratio between fighter stats
    print("Creating cross-feature derivatives...")
    
    # Dictionary of feature pairs to transform
    feature_pairs = {
        'AvgTD': ['RedAvgTDLanded', 'BlueAvgTDLanded'],
        'AvgSigStr': ['RedAvgSigStrLanded', 'BlueAvgSigStrLanded'],
        'WinStreak': ['RedCurrentWinStreak', 'BlueCurrentWinStreak'],
        'KOWins': ['RedWinsByKO', 'BlueWinsByKO'],
        'SubWins': ['RedWinsBySubmission', 'BlueWinsBySubmission'],
    }
    
    # Process each pair to create ratio features instead of diffs
    for name, (red_feat, blue_feat) in feature_pairs.items():
        if all(col in df_eng.columns for col in [red_feat, blue_feat]):
            # Create ratio features (handling zeros carefully)
            df_eng[f'{name}Ratio'] = df_eng[red_feat] / (df_eng[blue_feat].replace(0, 0.001))
            # Cap extreme ratios to prevent outlier influence
            df_eng[f'{name}Ratio'] = np.clip(df_eng[f'{name}Ratio'], 0.01, 100)
            # Apply log transformation to normalize the ratios
            df_eng[f'{name}LogRatio'] = np.log(df_eng[f'{name}Ratio'])
    
    # 3. Create interaction features between odds and performance metrics
    print("Creating strategic interaction features...")
    
    # Betting odds interaction with win records
    if all(col in df_eng.columns for col in ['RedOdds', 'RedWins']):
        df_eng['RedOddsPerWin'] = df_eng['RedOdds'] / np.maximum(1, df_eng['RedWins'])
    
    if all(col in df_eng.columns for col in ['BlueOdds', 'BlueWins']):
        df_eng['BlueOddsPerWin'] = df_eng['BlueOdds'] / np.maximum(1, df_eng['BlueWins'])
    
    # 4. Create non-linear transformations of key features
    print("Creating non-linear transformations...")
    
    # Identify highly skewed features
    skewed_features = ['RedCurrentWinStreak', 'BlueCurrentWinStreak', 'RedWins', 'BlueWins']
    for feature in skewed_features:
        if feature in df_eng.columns:
            # Log transform with offset to handle zeros
            df_eng[f'{feature}_log'] = np.log1p(df_eng[feature])
            # Square root transform
            df_eng[f'{feature}_sqrt'] = np.sqrt(df_eng[feature])
    
    # Enhanced features for finish prediction
    print("Creating specialized finish prediction features...")
    
    # 1. Grappling aggression indicators
    if all(col in df_eng.columns for col in ['RedAvgSubAtt', 'BlueAvgSubAtt']):
        df_eng['GrapplingAggression'] = df_eng['RedAvgSubAtt'] - df_eng['BlueAvgSubAtt']
        # Absolute differential - overall level of grappling activity
        df_eng['GrapplingActivity'] = np.abs(df_eng['RedAvgSubAtt']) + np.abs(df_eng['BlueAvgSubAtt'])
    
    # 2. Striking indicators
    if all(col in df_eng.columns for col in ['RedAvgSigStrLanded', 'BlueAvgSigStrLanded', 
                                            'RedAvgSigStrPct', 'BlueAvgSigStrPct']):
        # Striking effectiveness product
        df_eng['RedStrikingEffectiveness'] = df_eng['RedAvgSigStrLanded'] * df_eng['RedAvgSigStrPct']
        df_eng['BlueStrikingEffectiveness'] = df_eng['BlueAvgSigStrLanded'] * df_eng['BlueAvgSigStrPct']
        df_eng['StrikingEffectivenessDiff'] = df_eng['RedStrikingEffectiveness'] - df_eng['BlueStrikingEffectiveness']
    
    # 3. Finish history - strong indicator for future finishes
    if all(col in df_eng.columns for col in ['RedWinsByKO', 'RedWinsBySubmission', 'RedWins',
                                           'BlueWinsByKO', 'BlueWinsBySubmission', 'BlueWins']):
        # Calculate finish rates (adding small constant to avoid division by zero)
        df_eng['RedFinishRate'] = (df_eng['RedWinsByKO'] + df_eng['RedWinsBySubmission']) / np.maximum(df_eng['RedWins'], 1)
        df_eng['BlueFinishRate'] = (df_eng['BlueWinsByKO'] + df_eng['BlueWinsBySubmission']) / np.maximum(df_eng['BlueWins'], 1)
        # Combined finish propensity (likelihood either fighter gets a finish)
        df_eng['CombinedFinishRate'] = df_eng['RedFinishRate'] + df_eng['BlueFinishRate']
        # Difference in finish ability
        df_eng['FinishRateDiff'] = df_eng['RedFinishRate'] - df_eng['BlueFinishRate']
    
    # 4. Weight class features - useful since finish rates vary by weight class
    if 'RedWeightLbs' in df_eng.columns:
        # Create weight class buckets (approximate)
        def categorize_weight(weight):
            if weight <= 135:
                return 0  # Flyweight/Bantamweight
            elif weight <= 155:
                return 1  # Featherweight/Lightweight
            elif weight <= 185:
                return 2  # Welterweight/Middleweight
            else:
                return 3  # LHW/Heavyweight
                
        df_eng['WeightClassBucket'] = df_eng['RedWeightLbs'].apply(categorize_weight)
        
        # One-hot encode weight classes
        weight_dummies = pd.get_dummies(df_eng['WeightClassBucket'], prefix='WeightClass')
        df_eng = pd.concat([df_eng, weight_dummies], axis=1)
        df_eng.drop('WeightClassBucket', axis=1, inplace=True)
    
    # 5. Special feature: Fight Duration Expectation
    if 'NumberOfRounds' in df_eng.columns:
        # Create estimated fight duration (longer fights favor decisions)
        df_eng['ExpectedMaxDuration'] = df_eng['NumberOfRounds'] * 5 * 60  # in seconds
        
        # Create a 'title-fight' indicator
        df_eng['TitleFightInd'] = (df_eng['NumberOfRounds'] >= 5).astype(int)
    
    # 6. Style matchup features
    # Create product terms that capture when both fighters have similar strengths
    # These can indicate whether a fight is likely to be prolonged when styles clash
    if all(col in df_eng.columns for col in ['RedAvgTDLanded', 'BlueAvgTDLanded']):
        # Both fighters are heavy wrestlers - may lead to stalemates and decisions
        df_eng['WrestlingMatchup'] = df_eng['RedAvgTDLanded'] * df_eng['BlueAvgTDLanded']
    
    if all(col in df_eng.columns for col in ['RedAvgSigStrLanded', 'BlueAvgSigStrLanded']):
        # Both fighters are heavy strikers - may lead to finishes
        df_eng['StrikingMatchup'] = df_eng['RedAvgSigStrLanded'] * df_eng['BlueAvgSigStrLanded']
    
    # 7. Recent fight trends (time-series stats)
    print("Creating recent fight trend features...")
    
    # Create weighted momentum features based on win streak and finish types
    if all(col in df_eng.columns for col in ['RedCurrentWinStreak', 'BlueCurrentWinStreak', 
                                           'RedWinsByKO', 'RedWinsBySubmission',
                                           'BlueWinsByKO', 'BlueWinsBySubmission']):
        # Red corner momentum - weigh KO/TKO wins more heavily
        df_eng['RedFormMomentum'] = df_eng['RedCurrentWinStreak'] * (1 + 
            0.5 * (df_eng['RedWinsByKO'] + df_eng['RedWinsBySubmission']) / np.maximum(df_eng['RedWins'], 1))
        
        # Blue corner momentum
        df_eng['BlueFormMomentum'] = df_eng['BlueCurrentWinStreak'] * (1 + 
            0.5 * (df_eng['BlueWinsByKO'] + df_eng['BlueWinsBySubmission']) / np.maximum(df_eng['BlueWins'], 1))
        
        # Momentum differential (positive = red advantage, negative = blue advantage)
        df_eng['FormMomentumDiff'] = df_eng['RedFormMomentum'] - df_eng['BlueFormMomentum']
        
        # Combined momentum (high value = both fighters on strong streaks)
        df_eng['TotalMomentum'] = df_eng['RedFormMomentum'] + df_eng['BlueFormMomentum']
    
    # Create striking accuracy trend features
    # Note: This would ideally use actual recent fight data, but we'll approximate using available stats
    if all(col in df_eng.columns for col in ['RedAvgSigStrPct', 'BlueAvgSigStrPct']):
        # Red corner recent strike accuracy (approximated)
        # Adding random small variation to simulate trending data
        np.random.seed(42)  # For reproducibility
        recent_variation = np.random.uniform(-0.05, 0.05, size=len(df_eng))
        
        df_eng['RedRecentStrAcc'] = np.clip(df_eng['RedAvgSigStrPct'] + recent_variation, 0, 1)
        df_eng['BlueRecentStrAcc'] = np.clip(df_eng['BlueAvgSigStrPct'] + recent_variation, 0, 1)
        
        # Strike accuracy differential (recent trends)
        df_eng['RecentStrAccDiff'] = df_eng['RedRecentStrAcc'] - df_eng['BlueRecentStrAcc']
        
        # Strike accuracy improvement/decline (compared to career average)
        df_eng['RedStrAccTrend'] = df_eng['RedRecentStrAcc'] - df_eng['RedAvgSigStrPct']
        df_eng['BlueStrAccTrend'] = df_eng['BlueRecentStrAcc'] - df_eng['BlueAvgSigStrPct']
    
    # 8. Recent knockout power and submission prowess metrics
    if all(col in df_eng.columns for col in ['RedWinsByKO', 'RedWins', 'BlueWinsByKO', 'BlueWins']):
        # Knockout power ratio (higher = more KO wins in recent history)
        df_eng['RedRecentKOPower'] = np.power(df_eng['RedWinsByKO'] / np.maximum(df_eng['RedWins'], 1), 1.5)
        df_eng['BlueRecentKOPower'] = np.power(df_eng['BlueWinsByKO'] / np.maximum(df_eng['BlueWins'], 1), 1.5)
        
        # Combined KO power (fights with heavy hitters on both sides often end in finish)
        df_eng['CombinedKOPower'] = df_eng['RedRecentKOPower'] + df_eng['BlueRecentKOPower']
        
        # Submission prowess
        df_eng['RedRecentSubProwess'] = np.power(df_eng['RedWinsBySubmission'] / np.maximum(df_eng['RedWins'], 1), 1.5)
        df_eng['BlueRecentSubProwess'] = np.power(df_eng['BlueWinsBySubmission'] / np.maximum(df_eng['BlueWins'], 1), 1.5)
        
        # Combined submission threat
        df_eng['CombinedSubThreat'] = df_eng['RedRecentSubProwess'] + df_eng['BlueRecentSubProwess']
    
    # 9. Fight age/experience trends
    if 'Age' in df_eng.columns:
        # Age-momentum interaction (older fighters on win streaks are still dangerous)
        if all(col in df_eng.columns for col in ['RedAge', 'BlueAge', 'RedCurrentWinStreak', 'BlueCurrentWinStreak']):
            df_eng['RedAgeMomentum'] = df_eng['RedCurrentWinStreak'] / np.maximum(df_eng['RedAge'] - 25, 1)
            df_eng['BlueAgeMomentum'] = df_eng['BlueCurrentWinStreak'] / np.maximum(df_eng['BlueAge'] - 25, 1)
    
    print(f"Added {len(df_eng.columns) - len(df.columns)} new engineered features")
    return df_eng

def ensure_model_paths(use_calibration=True, use_stacking=True):
    """Ensure all required model directories exist and are properly linked
    
    Parameters:
    -----------
    use_calibration : bool, default=True
        If True, directory includes 'calibrated' in name
    use_stacking : bool, default=True
        If True, directory includes 'stacked_' in name
    """
    # Create all possible model directories
    os.makedirs('models', exist_ok=True)
    
    # Get absolute path of the models directory
    base_dir = os.path.abspath('models')
    
    # Current model directory (determined by calibration and stacking options)
    model_dir = 'models/'
    if use_stacking:
        model_dir += 'stacked_'
    model_dir += 'calibrated' if use_calibration else 'uncalibrated'
    
    # Make sure the specific model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Return the directory where models should be saved
    return model_dir, base_dir

def create_and_save_shap_explainers(winner_model, finish_model, X_train_winner, X_train_finish, winner_features, finish_features, model_dir):
    """
    Create and save SHAP explainers for the trained models
    """
    print("\nCreating SHAP explainers for model interpretability...")
    
    try:
        import shap
        
        # Create SHAP explainer for winner model
        print("Creating winner model explainer...")
        try:
            # Use a small sample of training data to speed up explainer creation
            sample_size = min(500, X_train_winner.shape[0])
            winner_background = shap.sample(X_train_winner, sample_size)
            
            if hasattr(winner_model, 'predict_proba'):
                def winner_predict_fn(x):
                    return winner_model.predict_proba(x)[:, 1]
                
                winner_explainer = shap.KernelExplainer(winner_predict_fn, winner_background)
                
                # Save the explainer
                print("Saving winner model explainer...")
                joblib.dump(winner_explainer, f'{model_dir}/winner_explainer.pkl')
                
                # Also save to the base models directory for easier access
                joblib.dump(winner_explainer, f'models/winner_explainer.pkl')
            else:
                print("Winner model doesn't support predict_proba, skipping explainer creation")
        except Exception as e:
            print(f"Error creating winner model explainer: {str(e)}")
        
        # Create SHAP explainer for finish model
        print("Creating finish model explainer...")
        try:
            # Use a small sample of training data to speed up explainer creation
            sample_size = min(500, X_train_finish.shape[0])
            finish_background = shap.sample(X_train_finish, sample_size)
            
            if hasattr(finish_model, 'predict_proba'):
                def finish_predict_fn(x):
                    return finish_model.predict_proba(x)[:, 1]
                
                finish_explainer = shap.KernelExplainer(finish_predict_fn, finish_background)
                
                # Save the explainer
                print("Saving finish model explainer...")
                joblib.dump(finish_explainer, f'{model_dir}/finish_explainer.pkl')
                
                # Also save to the base models directory for easier access
                joblib.dump(finish_explainer, f'models/finish_explainer.pkl')
            else:
                print("Finish model doesn't support predict_proba, skipping explainer creation")
        except Exception as e:
            print(f"Error creating finish model explainer: {str(e)}")
        
        # Save a small sample of training data for future explanation reference
        try:
            joblib.dump(winner_background, f'{model_dir}/background_data.pkl')
            joblib.dump(winner_background, f'models/background_data.pkl')
        except Exception as e:
            print(f"Error saving background data: {str(e)}")
            
    except ImportError:
        print("SHAP library not found. Install with: pip install shap")
        print("Skipping model explainer creation.")

def train_models(use_calibration=True, use_stacking=True):
    """
    Train UFC fight prediction models with improved probability calibration and ensemble stacking
    
    Parameters:
    -----------
    use_calibration : bool, default=True
        If True, applies CalibratedClassifierCV to improve probability estimates
    use_stacking : bool, default=True
        If True, uses a stacking ensemble to capture nonlinear patterns
    """
    print(f"Model calibration: {'Enabled' if use_calibration else 'Disabled'}")
    print(f"Ensemble stacking: {'Enabled' if use_stacking else 'Disabled'}")
    
    print("Loading data...")
    try:
        df = pd.read_csv("assets/ufc-master.csv")
    except FileNotFoundError:
        print("Error: Dataset file 'assets/ufc-master.csv' not found.")
        print("Please ensure the data file is in the correct location.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        sys.exit(1)
    
    print("Setting up directories...")
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/evaluation', exist_ok=True)
    
    start_time = datetime.now()
    print(f"Starting model training at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")    
    
    print("Preprocessing data...")
    rank_columns = [col for col in df.columns if 'weight' in col.lower() and 'rank' in col.lower()]
    for col in rank_columns:
        df[col] = df[col].fillna(0)
    
    df = df.dropna(subset=['RedOdds', 'BlueOdds'])
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col not in rank_columns]
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    df['BlueStance'] = df['BlueStance'].fillna('Orthodox')
    df['RedStance'] = df['RedStance'].fillna('Orthodox')
    df['FinishRoundTime'] = df['FinishRoundTime'].fillna('0:00')
    df['Finish'] = df['Finish'].fillna('Unknown')
    df['FinishDetails'] = df['FinishDetails'].fillna('Unknown')
    columns_to_drop = [
        'RedFighter', 'BlueFighter',
        'Date', 'Location', 'Country', 'EmptyArena'
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    df['WinnerNumeric'] = df['Winner'].apply(lambda x: 0 if x == 'Red' else 1)
    
    print("\nCreating finish type classification with 2 categories...")
    finish_mapping = {
        'KO/TKO': 0,
        'SUB': 0,
        'U-DEC': 1,
        'S-DEC': 1,
        'M-DEC': 1,  
        'DQ': 1,     
        'Unknown': 1,
        'Overturned': 1,
        'No Contest': 1
    }
    df['FinishNumeric'] = df['Finish'].map(finish_mapping)
    
    print("\nDistribution of finish types:")
    finish_counts = df['FinishNumeric'].value_counts()
    print(finish_counts)
    
    def convert_to_seconds(time_str):
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds
        except:
            return 0
    
    df['FinishRoundTimeSecs'] = df['FinishRoundTime'].apply(convert_to_seconds)
    leakage_columns = [
        'FinishRound',   
        'TotalFightTimeSecs',
        'FinishRoundTimeSecs',
        'Winner',
        'Finish',
        'FinishDetails'
    ]
    df = df.drop(columns=[col for col in leakage_columns if col in df.columns])
    
    joblib.dump(finish_mapping, 'models/finish_mapping.pkl')
    class_to_finish = {
        0: 'Finish (KO/TKO or Submission)',
        1: 'Decision'
    }
    joblib.dump(class_to_finish, 'models/class_to_finish.pkl')
    winner_class_names = ['Red Corner', 'Blue Corner']
    finish_class_names = ['Finish', 'Decision']
    
    # Apply enhanced feature engineering with a focus on finish prediction
    df = engineer_features(df)
    
    print("\nPreparing winner prediction model with Logistic Regression...")
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
    available_winner_features = [col for col in winner_features if col in df.columns]
    print(f"Using {len(available_winner_features)} winner features out of {len(winner_features)} requested")
    
    df_winner = df[available_winner_features + ['WinnerNumeric']]
    X_winner = df_winner.drop(columns=['WinnerNumeric'])
    y_winner = df_winner['WinnerNumeric']
    
    print("\nApplying paired feature selection for winner model...")
    selected_winner_features = paired_feature_selection(X_winner, y_winner, X_winner.columns, k=20)
    print(f"Selected {len(selected_winner_features)} features for winner model:")
    print(selected_winner_features)
    X_winner = X_winner[selected_winner_features]
    
    X_train_winner, X_test_winner, y_train_winner, y_test_winner = train_test_split(
        X_winner, y_winner, test_size=0.2, random_state=42, stratify=y_winner
    )
    winner_scaler = StandardScaler()
    X_train_winner_scaled = winner_scaler.fit_transform(X_train_winner)
    X_test_winner_scaled = winner_scaler.transform(X_test_winner)

    print("\nTraining models for winner prediction...")
    
    # Use predefined best parameters for winner model
    winner_params = {
        'C': 0.1, 
        'class_weight': 'balanced', 
        'l1_ratio': 1, 
        'max_iter': 10000, 
        'penalty': 'elasticnet', 
        'solver': 'saga'
    }
    
    # Create base logistic regression model
    base_winner_lr = LogisticRegression(**winner_params)
    if use_stacking:
        # Create a stacking ensemble to capture nonlinear patterns
        print("Creating stacking ensemble with LogisticRegression and GradientBoosting...")
        
        # Define base estimators for stacking
        gb_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
        
        base_estimators = [
            ('lr', LogisticRegression(**winner_params)),
            ('gb', GradientBoostingClassifier(**gb_params))
        ]
        
        # Create the stacking model
        base_winner_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(class_weight='balanced'),
            cv=5
        )
        print("Using stacked ensemble as base winner model")
    else:
        # Use simple logistic regression
        base_winner_model = base_winner_lr
        print("Using Logistic Regression as base winner model")
    
    # Apply calibration if requested
    if use_calibration:
        print("Applying probability calibration to winner model...")
        winner_model = CalibratedClassifierCV(
            base_winner_model,
            cv=5,
            method='isotonic'
        )
    else:
        winner_model = base_winner_model
    
    winner_model.fit(X_train_winner_scaled, y_train_winner)
    winner_pred = winner_model.predict(X_test_winner_scaled)
    winner_proba = winner_model.predict_proba(X_test_winner_scaled)
    
    # Calculate additional probabilistic metrics
    winner_accuracy = accuracy_score(y_test_winner, winner_pred)
    winner_auc = roc_auc_score(y_test_winner, winner_proba[:, 1])
    winner_log_loss = log_loss(y_test_winner, winner_proba)
    winner_brier = brier_score_loss(y_test_winner, winner_proba[:, 1])
    
    print(f"\nWinner prediction model")
    print(f"Accuracy: {winner_accuracy:.4f}")
    print(f"AUC-ROC: {winner_auc:.4f}")
    print(f"Log Loss: {winner_log_loss:.4f}")
    print(f"Brier Score: {winner_brier:.4f} (lower is better)")
    print("\nClassification Report:")
    print(classification_report(y_test_winner, winner_pred))
    
    print("\nPreparing finish prediction model...")
    # Prepare finish prediction model features
    finish_features = [
        # Basic features
        'RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue',
        'TitleBout', 'NumberOfRounds', 'RedCurrentLoseStreak', 'BlueCurrentLoseStreak',
        'RedCurrentWinStreak', 'BlueCurrentWinStreak', 'RedAvgSigStrLanded', 'BlueAvgSigStrLanded',
        'RedAvgSigStrPct', 'BlueAvgSigStrPct', 'RedAvgSubAtt', 'BlueAvgSubAtt',
        'RedAvgTDLanded', 'BlueAvgTDLanded', 'RedWinsByKO', 'BlueWinsByKO',
        'RedWinsBySubmission', 'BlueWinsBySubmission', 'RedWinsByDecisionUnanimous',
        'BlueWinsByDecisionUnanimous', 'RedWins', 'BlueWins', 'RedHeightCms',
        'BlueHeightCms', 'RedReachCms', 'BlueReachCms', 'RedWeightLbs', 'BlueWeightLbs',
        'WinStreakDif', 'LoseStreakDif', 'HeightDif', 'ReachDif', 'AgeDif', 'SigStrDif',
        'AvgSubAttDif', 'AvgTDDif',
        
        # New engineered features specifically for finish prediction
        'GrapplingAggression', 'GrapplingActivity',
        'RedStrikingEffectiveness', 'BlueStrikingEffectiveness', 'StrikingEffectivenessDiff',
        'RedFinishRate', 'BlueFinishRate', 'CombinedFinishRate', 'FinishRateDiff',
        'ExpectedMaxDuration', 'TitleFightInd', 'WrestlingMatchup', 'StrikingMatchup',
        'LogOddsRatio', 'AvgTDLogRatio', 'AvgSigStrLogRatio', 'WinStreakLogRatio', 
        'KOWinsLogRatio', 'SubWinsLogRatio', 'RedOddsPerWin', 'BlueOddsPerWin',
        'WeightClass_0', 'WeightClass_1', 'WeightClass_2', 'WeightClass_3',
        
        # Recent fight trend features
        'RedFormMomentum', 'BlueFormMomentum', 'FormMomentumDiff', 'TotalMomentum',
        'RedRecentStrAcc', 'BlueRecentStrAcc', 'RecentStrAccDiff',
        'RedStrAccTrend', 'BlueStrAccTrend',
        'RedRecentKOPower', 'BlueRecentKOPower', 'CombinedKOPower',
        'RedRecentSubProwess', 'BlueRecentSubProwess', 'CombinedSubThreat',
        'RedAgeMomentum', 'BlueAgeMomentum'
    ]
    
    # Get all available features from the engineered dataframe
    available_finish_features = [col for col in finish_features if col in df.columns]
    print(f"Using {len(available_finish_features)} finish features out of {len(finish_features)} requested")
    k_finish_features = 30  # Increased from 20 to capture more features
    
    df_finish = df[available_finish_features + ['FinishNumeric']]
    X_finish = df_finish.drop(columns=['FinishNumeric'])
    y_finish = df_finish['FinishNumeric']
    
    print("\nApplying paired feature selection for finish model with expanded feature set...")
    selected_finish_features = paired_feature_selection(X_finish, y_finish, X_finish.columns, k=k_finish_features)
    print(f"Selected {len(selected_finish_features)} features for finish model:")
    print(selected_finish_features)
    
    X_finish = X_finish[selected_finish_features]
    X_train_finish, X_test_finish, y_train_finish, y_test_finish = train_test_split(
        X_finish, y_finish, test_size=0.2, random_state=42, stratify=y_finish
    )
    finish_scaler = StandardScaler()
    X_train_finish_scaled = finish_scaler.fit_transform(X_train_finish)
    X_test_finish_scaled = finish_scaler.transform(X_test_finish)
    
    print("\nTraining models for finish prediction...")
    # Use predefined best parameters for finish model
    finish_params = {
        'C': 0.01, 
        'class_weight': 'balanced', 
        'max_iter': 10000, 
        'penalty': 'l2', 
        'solver': 'lbfgs'
    }
    
    # Create base logistic regression model
    base_finish_lr = LogisticRegression(**finish_params)
    
    if use_stacking:
        # Create a stacking ensemble to capture nonlinear patterns
        print("Creating stacking ensemble for finish prediction...")
        
        # Define base estimators for stacking - can tune specifically for finish task
        gb_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 4,  # Slightly more complex for finish prediction
            'random_state': 42
        }
        
        base_estimators = [
            ('lr', LogisticRegression(**finish_params)),
            ('gb', GradientBoostingClassifier(**gb_params))
        ]
        
        # Create the stacking model
        base_finish_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(class_weight='balanced'),
            cv=5
        )
        print("Using stacked ensemble as base finish model")
    else:
        # Use simple logistic regression
        base_finish_model = base_finish_lr
        print("Using Logistic Regression as base finish model")
    
    # Apply calibration if requested
    if use_calibration:
        print("Applying probability calibration to finish model...")
        finish_model = CalibratedClassifierCV(
            base_finish_model,
            cv=5,
            method='isotonic'
        )
    else:
        finish_model = base_finish_model
    
    finish_model.fit(X_train_finish_scaled, y_train_finish)
    finish_pred = finish_model.predict(X_test_finish_scaled)
    finish_proba = finish_model.predict_proba(X_test_finish_scaled)
    
    # Calculate additional probabilistic metrics for finish model
    finish_accuracy = accuracy_score(y_test_finish, finish_pred)
    finish_auc = roc_auc_score(y_test_finish, finish_proba[:, 1])
    finish_log_loss = log_loss(y_test_finish, finish_proba)
    finish_brier = brier_score_loss(y_test_finish, finish_proba[:, 1])
    
    print(f"\nFinish prediction model")
    print(f"Accuracy: {finish_accuracy:.4f}")
    print(f"AUC-ROC: {finish_auc:.4f}")
    print(f"Log Loss: {finish_log_loss:.4f}")
    print(f"Brier Score: {finish_brier:.4f} (lower is better)")
    print("\nClassification Report:")
    print(classification_report(y_test_finish, finish_pred))
    
    print("\nSaving models and scalers...")
    model_dir, base_dir = ensure_model_paths(use_calibration, use_stacking)
    
    # Save models in their specific directory
    joblib.dump(winner_model, f'{model_dir}/winner_model.pkl')
    joblib.dump(finish_model, f'{model_dir}/finish_model.pkl')
    joblib.dump(winner_scaler, f'{model_dir}/winner_scaler.pkl')
    joblib.dump(finish_scaler, f'{model_dir}/finish_scaler.pkl')
    joblib.dump(selected_winner_features, f'{model_dir}/winner_features.pkl')
    joblib.dump(selected_finish_features, f'{model_dir}/finish_features.pkl')
    
    # Also save to base models directory for easier discovery
    print("Creating copies in base models directory for easier access...")
    joblib.dump(winner_model, 'models/winner_model.pkl')
    joblib.dump(finish_model, 'models/finish_model.pkl')
    joblib.dump(winner_scaler, 'models/winner_scaler.pkl')
    joblib.dump(finish_scaler, 'models/finish_scaler.pkl')
    joblib.dump(selected_winner_features, 'models/winner_features.pkl')
    joblib.dump(selected_finish_features, 'models/finish_features.pkl')
    
    print("\nTraining complete. Models and scalers saved.")
    
    # Create and save SHAP explainers if possible
    try:
        create_and_save_shap_explainers(
            winner_model, 
            finish_model, 
            X_train_winner_scaled, 
            X_train_finish_scaled,
            selected_winner_features,
            selected_finish_features,
            model_dir
        )
    except Exception as e:
        print(f"Could not create SHAP explainers: {str(e)}")
    
    return {
        'winner_accuracy': winner_accuracy,
        'finish_accuracy': finish_accuracy
    }

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRAINING UFC FIGHT PREDICTION MODELS")
    print("="*80)
    train_models(use_calibration=True, use_stacking=True)
