from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_and_preprocess_data(input_data):
    df = pd.DataFrame(input_data)
    
    # Remove outliers using IQR for relevant features
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    return df

def scale_features(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    return scaled_features, scaler

def preprocess_winner_data(input_data):
    df = load_and_preprocess_data(input_data)
    # Ensure we only use the features that the winner model was trained on
    winner_features = ['RedOdds', 'BlueOdds', 'RedAvgSigStrLanded', 'BlueAvgSigStrLanded', 
                   'RedCurrentWinStreak', 'BlueCurrentWinStreak', 'RedHeightCms', 
                   'BlueHeightCms', 'RedReachCms', 'BlueReachCms']
    
    # Make sure all required columns exist
    for col in winner_features:
        if col not in df.columns:
            df[col] = 0  # Default to 0 for missing columns
            
    features = df[winner_features]
    scaled_features, scaler = scale_features(features)
    return scaled_features, scaler

def preprocess_finish_data(input_data):
    df = load_and_preprocess_data(input_data)
    # Ensure we only use the features that the finish model was trained on
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