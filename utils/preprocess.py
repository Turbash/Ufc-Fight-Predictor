import joblib
import pandas as pd
import numpy as np

def preprocess_winner_data(input_data):
    df = load_and_preprocess_data(input_data)
    
    # Load the exact features the winner model was trained on
    try:
        winner_features = joblib.load('models/winner_features.pkl')
    except:
        # Fall back to a default list of features if the file is not found
        winner_features = ['RedOdds', 'BlueOdds', 'RedAvgSigStrLanded', 'BlueAvgSigStrLanded', 
                      'RedCurrentWinStreak', 'BlueCurrentWinStreak', 'RedHeightCms', 
                      'BlueHeightCms', 'RedReachCms', 'BlueReachCms',
                      'RedAvgSigStrPct', 'BlueAvgSigStrPct', 'RedAvgSubAtt', 'BlueAvgSubAtt',
                      'RedAvgTDLanded', 'BlueAvgTDLanded', 'RedAvgTDPct', 'BlueAvgTDPct',
                      'RedWins', 'BlueWins', 'RedLosses', 'BlueLosses',
                      'SigStrDif', 'HeightDif', 'ReachDif', 'AgeDif', 'AvgSubAttDif',
                      'AvgTDDif', 'WinStreakDif', 'LoseStreakDif', 'TotalRoundDif',
                      'RedAge', 'BlueAge', 'RedWeightLbs', 'BlueWeightLbs', 
                      'RedTotalRoundsFought', 'BlueTotalRoundsFought',
                      'RedWinsByKO', 'BlueWinsByKO']
    
    # Make sure all required columns exist
    for col in winner_features:
        if col not in df.columns:
            df[col] = 0  # Default to 0 for missing columns
    
    # Select only the columns the model was trained on
    df_selected = df[winner_features].copy()
    
    # Apply scaling using the same scaler used during training
    try:
        # Load the scaler used during training
        winner_scaler = joblib.load('models/winner_scaler.pkl')
        df_scaled = winner_scaler.transform(df_selected)
        return df_scaled
    except:
        # If scaler isn't found, return unscaled data with a warning
        print("Warning: Winner scaler not found. Using unscaled data.")
        return df_selected

def preprocess_finish_data(input_data):
    df = load_and_preprocess_data(input_data)
    
    # Load the exact features the finish model was trained on
    try:
        finish_features = joblib.load('models/finish_features.pkl')
    except:
        # If we can't load the features, use a default set based on training script
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
    
    # Create a DataFrame with just the required features
    finish_data = pd.DataFrame(index=[0])
    for feature in finish_features:
        if feature in df.columns:
            finish_data[feature] = df[feature]
        else:
            finish_data[feature] = 0
    
    # Apply scaling using the same scaler used during training
    try:
        # Load the scaler used during training
        finish_scaler = joblib.load('models/finish_scaler.pkl')
        finish_data_scaled = finish_scaler.transform(finish_data)
        return finish_data_scaled
    except:
        # If scaler isn't found, return unscaled data with a warning
        print("Warning: Finish scaler not found. Using unscaled data.")
        return finish_data

# Add wrapper functions to match the imports in __init__.py
def preprocess_winner_input(input_data):
    """
    Wrapper function for preprocess_winner_data to maintain compatibility
    """
    return preprocess_winner_data(input_data)
    
def preprocess_finish_input(input_data):
    """
    Wrapper function for preprocess_finish_data to maintain compatibility
    """
    return preprocess_finish_data(input_data)

def load_and_preprocess_data(input_data):
    """
    Convert input data to DataFrame and perform any necessary preprocessing
    """
    if isinstance(input_data, pd.DataFrame):
        return input_data
    elif isinstance(input_data, np.ndarray):
        return pd.DataFrame(input_data)
    else:
        # Handle other types or raise error
        raise ValueError("Input data must be DataFrame or numpy array")