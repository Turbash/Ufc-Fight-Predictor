import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_features(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    return scaled_features, scaler

def preprocess_winner_data(input_data):
    df = load_and_preprocess_data(input_data)
    
    # Define the exact 39 features to use (matching what the model was trained on)
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
    # Note: Removed 'WinDif' from the feature list to match the 39 features model expects
    
    # Make sure all required columns exist
    for col in winner_features:
        if col not in df.columns:
            df[col] = 0  # Default to 0 for missing columns
    
    # Add a safety check: ensure we ONLY use the 39 columns the model was trained on
    # This prevents the "X has N features but model expects M features" error
    df_selected = df[winner_features].copy()
    
    # Double check feature count
    expected_feature_count = 39
    actual_feature_count = df_selected.shape[1]
    
    if actual_feature_count != expected_feature_count:
        raise ValueError(f"Feature count mismatch: Expected {expected_feature_count} but got {actual_feature_count}")
    
    # Scale the features
    scaled_features, scaler = scale_features(df_selected)
    
    # Final verification
    if scaled_features.shape[1] != expected_feature_count:
        raise ValueError(f"Scaled feature count mismatch: Expected {expected_feature_count} but got {scaled_features.shape[1]}")
    
    return scaled_features, scaler

def preprocess_finish_data(input_data):
    df = load_and_preprocess_data(input_data)
    
    # Load the exact features the finish model was trained on
    try:
        finish_features = joblib.load('models/finish_features.pkl')
    except:
        # Fallback to default features
        finish_features = ['RedAvgSigStrLanded', 'BlueAvgSigStrLanded', 'RedAvgTDLanded', 
                      'BlueAvgTDLanded', 'RedAvgSubAtt', 'BlueAvgSubAtt', 
                      'RedCurrentWinStreak', 'BlueCurrentWinStreak']
    
    # Make sure all required columns exist
    for col in finish_features:
        if col not in df.columns:
            df[col] = 0  # Default to 0 for missing columns
            
    features = df[finish_features]
    scaled_features, scaler = scale_features(features)
    return scaled_features, scaler

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