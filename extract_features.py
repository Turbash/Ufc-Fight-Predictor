import pandas as pd
import numpy as np
import joblib
import os

def extract_all_features_from_csv(df):
    """
    Extract and preprocess all features from a DataFrame of fights
    to make them ready for model prediction.
    """
    # Create a new DataFrame with properly processed features
    processed_df = pd.DataFrame()
    
    # Check if input is empty
    if df is None or df.empty:
        print("Warning: Empty DataFrame provided to extract_all_features_from_csv")
        return pd.DataFrame()
    
    # Copy over basic fight information if available
    for col in ['RedFighter', 'BlueFighter', 'WeightClass']:
        if col in df.columns:
            processed_df[col] = df[col]
        else:
            # Generate default values for missing columns
            if col == 'RedFighter':
                processed_df[col] = ['Red Fighter' + str(i) for i in range(len(df))]
            elif col == 'BlueFighter':
                processed_df[col] = ['Blue Fighter' + str(i) for i in range(len(df))]
            elif col == 'WeightClass':
                processed_df[col] = ['Unknown' for _ in range(len(df))]
    
    # Process odds and values
    for col in ['RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue']:
        if col in df.columns:
            processed_df[col] = df[col].astype(float)
        else:
            # Default values for missing columns
            if col in ['RedExpectedValue', 'BlueExpectedValue']:
                # Calculate expected values from odds if available
                if 'RedOdds' in df.columns and 'BlueOdds' in df.columns and col == 'RedExpectedValue':
                    red_odds = df['RedOdds'].astype(float)
                    processed_df[col] = calculate_expected_value(red_odds)
                elif 'RedOdds' in df.columns and 'BlueOdds' in df.columns and col == 'BlueExpectedValue':
                    blue_odds = df['BlueOdds'].astype(float)
                    processed_df[col] = calculate_expected_value(blue_odds)
                else:
                    processed_df[col] = 0.5  # Default to 50% probability
            else:
                processed_df[col] = 0.0  # Default for missing odds
    
    # Process numerical stats - all potentially needed features
    numerical_cols = [
        'RedAge', 'BlueAge', 'RedCurrentWinStreak', 'BlueCurrentWinStreak',
        'RedCurrentLoseStreak', 'BlueCurrentLoseStreak', 'RedAvgSigStrLanded',
        'BlueAvgSigStrLanded', 'RedAvgSigStrPct', 'BlueAvgSigStrPct',
        'RedAvgSubAtt', 'BlueAvgSubAtt', 'RedAvgTDLanded', 'BlueAvgTDLanded', 
        'RedAvgTDPct', 'BlueAvgTDPct', 'RedHeightCms', 'BlueHeightCms', 
        'RedReachCms', 'BlueReachCms', 'RedWeightLbs', 'BlueWeightLbs', 
        'NumberOfRounds', 'TitleBout', 'RedTotalRoundsFought', 
        'BlueTotalRoundsFought', 'RedDecOdds', 'BlueDecOdds', 
        'RSubOdds', 'BSubOdds', 'RKOOdds', 'BKOOdds'
    ]
    
    for col in numerical_cols:
        if col in df.columns:
            # Convert percentage values to decimals if needed (e.g., 45% -> 0.45)
            if col in ['RedAvgSigStrPct', 'BlueAvgSigStrPct', 'RedAvgTDPct', 'BlueAvgTDPct']:
                # Check if values are already in decimal format (0-1)
                if df[col].max() > 1.0:
                    processed_df[col] = df[col].fillna(0).astype(float) / 100
                else:
                    processed_df[col] = df[col].fillna(0).astype(float)
            else:
                processed_df[col] = df[col].fillna(0).astype(float)
        else:
            processed_df[col] = 0.0  # Default for missing columns
    
    # Process win method columns
    win_method_cols = [
        'RedWins', 'BlueWins', 'RedLosses', 'BlueLosses',
        'RedWinsByKO', 'BlueWinsByKO', 'RedWinsBySubmission', 
        'BlueWinsBySubmission', 'RedWinsByDecisionUnanimous',
        'BlueWinsByDecisionUnanimous'
    ]
    
    for col in win_method_cols:
        if col in df.columns:
            processed_df[col] = df[col].fillna(0).astype(int)
        else:
            # For total wins/losses, try to calculate from components if possible
            if col == 'RedWins' and all(x in df.columns for x in ['RedWinsByKO', 'RedWinsBySubmission', 'RedWinsByDecisionUnanimous']):
                processed_df[col] = df['RedWinsByKO'].fillna(0).astype(int) + \
                                   df['RedWinsBySubmission'].fillna(0).astype(int) + \
                                   df['RedWinsByDecisionUnanimous'].fillna(0).astype(int)
            elif col == 'BlueWins' and all(x in df.columns for x in ['BlueWinsByKO', 'BlueWinsBySubmission', 'BlueWinsByDecisionUnanimous']):
                processed_df[col] = df['BlueWinsByKO'].fillna(0).astype(int) + \
                                   df['BlueWinsBySubmission'].fillna(0).astype(int) + \
                                   df['BlueWinsByDecisionUnanimous'].fillna(0).astype(int)
            else:
                processed_df[col] = 0  # Default for missing columns
    
    # Calculate differential features
    dif_calculations = {
        'WinStreakDif': ('RedCurrentWinStreak', 'BlueCurrentWinStreak'),
        'LoseStreakDif': ('RedCurrentLoseStreak', 'BlueCurrentLoseStreak'),
        'HeightDif': ('RedHeightCms', 'BlueHeightCms'),
        'ReachDif': ('RedReachCms', 'BlueReachCms'),
        'AgeDif': ('RedAge', 'BlueAge'),
        'SigStrDif': ('RedAvgSigStrLanded', 'BlueAvgSigStrLanded'),
        'AvgSubAttDif': ('RedAvgSubAtt', 'BlueAvgSubAtt'),
        'AvgTDDif': ('RedAvgTDLanded', 'BlueAvgTDLanded'),
        'WinDif': ('RedWins', 'BlueWins'),
        'LossDif': ('RedLosses', 'BlueLosses'),
        'TotalRoundDif': ('RedTotalRoundsFought', 'BlueTotalRoundsFought')
    }
    
    # Ensure all dif columns are calculated or set to 0
    for dif_name, (red_col, blue_col) in dif_calculations.items():
        if red_col in processed_df.columns and blue_col in processed_df.columns:
            processed_df[dif_name] = processed_df[red_col] - processed_df[blue_col]
        else:
            processed_df[dif_name] = 0.0
    
    # Calculate advanced derived features used by the models
    try:
        from utils.preprocess import calculate_derived_features
        processed_df = calculate_derived_features(processed_df)
    except:
        print("Warning: Could not calculate advanced derived features")
    
    return processed_df

def calculate_expected_value(odds):
    """Calculate implied probability from betting odds"""
    # Convert odds to probabilities
    # For negative odds (favorites): abs(odds) / (abs(odds) + 100)
    # For positive odds (underdogs): 100 / (odds + 100)
    result = []
    for odd in odds:
        if odd < 0:  # Favorite
            result.append(abs(odd) / (abs(odd) + 100))
        else:  # Underdog
            result.append(100 / (odd + 100))
    return result

def save_model_features():
    """
    Extract and save the feature sets used by the models for consistent prediction.
    """
    try:
        # Check if models directory exists, if not create it
        os.makedirs('models', exist_ok=True)
        
        # Check if models are already saved
        if not os.path.exists('models/winner_model.pkl') or not os.path.exists('models/finish_model.pkl'):
            print("Models not found. Please run train_models.py first.")
            return False
        
        # Load models - both should be LogisticRegression models
        winner_model = joblib.load('models/winner_model.pkl')
        finish_model = joblib.load('models/finish_model.pkl')
        
        # Check if feature list files already exist (from training)
        if os.path.exists('models/winner_features.pkl') and os.path.exists('models/finish_features.pkl'):
            print("Feature files already exist - using those for consistency.")
            return True
        
        # If feature files don't exist, try to extract from model or create based on model parameters
        # For Logistic Regression, we need to check for feature names saved during training
        
        # Default basic feature sets in case extraction fails
        default_winner_features = ['RedOdds', 'BlueOdds', 'RedCurrentWinStreak', 
                                  'BlueCurrentWinStreak', 'RedAvgSigStrLanded', 
                                  'BlueAvgSigStrLanded', 'WinStreakDif', 'HeightDif']
                                  
        default_finish_features = ['RedOdds', 'BlueOdds', 'NumberOfRounds', 
                                  'RedWinsByKO', 'BlueWinsByKO', 'RedWinsBySubmission', 
                                  'BlueWinsBySubmission']
        
        # Save the feature lists - using defaults if extraction failed
        joblib.dump(default_winner_features, 'models/winner_features.pkl')
        joblib.dump(default_finish_features, 'models/finish_features.pkl')
        
        print("Model features saved (using default feature sets)")
        return True
    except Exception as e:
        print(f"Error saving model features: {str(e)}")
        return False

if __name__ == "__main__":
    # Run this to save feature lists for both models
    save_model_features()
