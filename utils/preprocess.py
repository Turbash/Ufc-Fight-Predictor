import joblib
import pandas as pd
import numpy as np
import os

def preprocess_winner_data(input_data):
    # Load and preprocess data with calculated derived features
    df = load_and_preprocess_data(input_data)
    
    # Calculate any needed derived features
    df = calculate_derived_features(df)
    
    # Load the exact features the winner model was trained on
    try:
        winner_features = joblib.load('models/winner_features.pkl')
    except Exception as e:
        # Fall back to a basic set of features - using only the core features
        print(f"Warning: Could not load winner features: {str(e)}")
        winner_features = ['RedOdds', 'BlueOdds', 'RedCurrentWinStreak', 'BlueCurrentWinStreak',
                           'RedAvgSigStrLanded', 'BlueAvgSigStrLanded', 'WinStreakDif',
                           'HeightDif', 'ReachDif', 'AgeDif']
    
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
    except Exception as e:
        # If scaler isn't found, return unscaled data with a warning
        print(f"Warning: Winner scaler not found or error applying scaling: {str(e)}. Using unscaled data.")
        return df_selected

def preprocess_finish_data(input_data):
    # Load and preprocess data with calculated derived features
    df = load_and_preprocess_data(input_data)
    
    # Calculate any needed derived features
    df = calculate_derived_features(df)
    
    # Load the exact features the finish model was trained on
    try:
        finish_features = joblib.load('models/finish_features.pkl')
    except Exception as e:
        # Fall back to a basic set of features - using only core features
        print(f"Warning: Could not load finish features: {str(e)}")
        finish_features = ['RedOdds', 'BlueOdds', 'NumberOfRounds', 'TitleBout',
                           'RedWinsByKO', 'BlueWinsByKO', 'RedAvgSigStrLanded',
                           'BlueAvgSigStrLanded', 'SigStrDif']
    
    # Make sure all required columns exist
    for col in finish_features:
        if col not in df.columns:
            df[col] = 0  # Default to 0 for missing columns
    
    # Select only the columns the model was trained on
    df_selected = df[finish_features].copy()
    
    # Apply scaling using the same scaler used during training
    try:
        # Load the scaler used during training
        finish_scaler = joblib.load('models/finish_scaler.pkl')
        df_scaled = finish_scaler.transform(df_selected)
        return df_scaled
    except Exception as e:
        # If scaler isn't found, return unscaled data with a warning
        print(f"Warning: Finish scaler not found or error applying scaling: {str(e)}. Using unscaled data.")
        return df_selected

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
        return input_data.copy()  # Return a copy to avoid modifying the original
    elif isinstance(input_data, np.ndarray):
        return pd.DataFrame(input_data)
    elif isinstance(input_data, dict):
        # Handle dict input by converting to DataFrame
        return pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        # Handle list input by converting to DataFrame
        return pd.DataFrame(input_data)
    else:
        # Handle other types or raise error
        raise ValueError("Input data must be DataFrame, numpy array, dict or list")

def calculate_derived_features(df):
    """
    Calculate all the derived features needed by the models based on the input features
    """
    # Create a copy to avoid modifying the original
    df_derived = df.copy()
    
    # Calculate differentials
    diff_pairs = {
        'WinStreakDif': ('RedCurrentWinStreak', 'BlueCurrentWinStreak'),
        'LoseStreakDif': ('RedCurrentLoseStreak', 'BlueCurrentLoseStreak'),
        'HeightDif': ('RedHeightCms', 'BlueHeightCms'),
        'ReachDif': ('RedReachCms', 'BlueReachCms'),
        'AgeDif': ('RedAge', 'BlueAge'),
        'SigStrDif': ('RedAvgSigStrLanded', 'BlueAvgSigStrLanded'),
        'AvgSubAttDif': ('RedAvgSubAtt', 'BlueAvgSubAtt'),
        'AvgTDDif': ('RedAvgTDLanded', 'BlueAvgTDLanded')
    }
    
    for diff_name, (red_col, blue_col) in diff_pairs.items():
        if red_col in df.columns and blue_col in df.columns:
            df_derived[diff_name] = df[red_col] - df[blue_col]
    
    # Calculate striking effectiveness if needed
    if all(col in df.columns for col in ['RedAvgSigStrLanded', 'BlueAvgSigStrLanded', 
                                        'RedAvgSigStrPct', 'BlueAvgSigStrPct']):
        # Striking effectiveness product
        df_derived['RedStrikingEffectiveness'] = df['RedAvgSigStrLanded'] * df['RedAvgSigStrPct']
        df_derived['BlueStrikingEffectiveness'] = df['BlueAvgSigStrLanded'] * df['BlueAvgSigStrPct']
        df_derived['StrikingEffectivenessDiff'] = df_derived['RedStrikingEffectiveness'] - df_derived['BlueStrikingEffectiveness']
        df_derived['StrikingMatchup'] = df['RedAvgSigStrLanded'] * df['BlueAvgSigStrLanded']
    
    # Calculate grappling metrics if needed
    if all(col in df.columns for col in ['RedAvgSubAtt', 'BlueAvgSubAtt']):
        df_derived['GrapplingAggression'] = df['RedAvgSubAtt'] - df['BlueAvgSubAtt']
        df_derived['GrapplingActivity'] = df['RedAvgSubAtt'] + df['BlueAvgSubAtt']
    
    if all(col in df.columns for col in ['RedAvgTDLanded', 'BlueAvgTDLanded']):
        df_derived['WrestlingMatchup'] = df['RedAvgTDLanded'] * df['BlueAvgTDLanded']
    
    # Calculate finish-related features if needed
    if all(col in df.columns for col in ['RedWinsByKO', 'RedWinsBySubmission', 'RedWins']):
        df_derived['RedFinishRate'] = (df['RedWinsByKO'] + df['RedWinsBySubmission']) / np.maximum(df['RedWins'], 1)
    
    if all(col in df.columns for col in ['BlueWinsByKO', 'BlueWinsBySubmission', 'BlueWins']):
        df_derived['BlueFinishRate'] = (df['BlueWinsByKO'] + df['BlueWinsBySubmission']) / np.maximum(df['BlueWins'], 1)
    
    if all(col in df_derived.columns for col in ['RedFinishRate', 'BlueFinishRate']):
        df_derived['CombinedFinishRate'] = df_derived['RedFinishRate'] + df_derived['BlueFinishRate']
        df_derived['FinishRateDiff'] = df_derived['RedFinishRate'] - df_derived['BlueFinishRate']
    
    # Log odds ratio if betting odds are available
    if all(col in df.columns for col in ['RedOdds', 'BlueOdds']):
        # Handle zero and negative odds
        red_odds = np.abs(df['RedOdds'])
        blue_odds = np.abs(df['BlueOdds'])
        
        # Add small constant to avoid division by zero
        red_odds = np.maximum(red_odds, 1e-5)
        blue_odds = np.maximum(blue_odds, 1e-5)
        
        df_derived['LogOddsRatio'] = np.log(red_odds / blue_odds)
    
    # Calculate fight duration expectation
    if 'NumberOfRounds' in df.columns:
        df_derived['ExpectedMaxDuration'] = df['NumberOfRounds'] * 5 * 60  # in seconds
        df_derived['TitleFightInd'] = (df['NumberOfRounds'] >= 5).astype(int)
    
    # Calculate momentum features if needed
    if all(col in df.columns for col in ['RedCurrentWinStreak', 'BlueCurrentWinStreak']):
        # Basic momentum without finish weights
        df_derived['FormMomentumDiff'] = df['RedCurrentWinStreak'] - df['BlueCurrentWinStreak']
        df_derived['TotalMomentum'] = df['RedCurrentWinStreak'] + df['BlueCurrentWinStreak']
        
        # Calculate more complex momentum with finish weights if data available
        if all(col in df.columns for col in ['RedWinsByKO', 'RedWinsBySubmission', 'BlueWinsByKO', 'BlueWinsBySubmission']):
            red_finish_bonus = 0.5 * (df['RedWinsByKO'] + df['RedWinsBySubmission']) / np.maximum(df['RedWins'], 1)
            df_derived['RedFormMomentum'] = df['RedCurrentWinStreak'] * (1 + red_finish_bonus)
            
            blue_finish_bonus = 0.5 * (df['BlueWinsByKO'] + df['BlueWinsBySubmission']) / np.maximum(df['BlueWins'], 1)
            df_derived['BlueFormMomentum'] = df['BlueCurrentWinStreak'] * (1 + blue_finish_bonus)
            
            # Update momentum differential and total with weighted versions
            df_derived['FormMomentumDiff'] = df_derived['RedFormMomentum'] - df_derived['BlueFormMomentum']
            df_derived['TotalMomentum'] = df_derived['RedFormMomentum'] + df_derived['BlueFormMomentum']
    
    # KO power metrics
    if all(col in df.columns for col in ['RedWinsByKO', 'RedWins', 'BlueWinsByKO', 'BlueWins']):
        df_derived['RedRecentKOPower'] = np.power(df['RedWinsByKO'] / np.maximum(df['RedWins'], 1), 1.5)
        df_derived['BlueRecentKOPower'] = np.power(df['BlueWinsByKO'] / np.maximum(df['BlueWins'], 1), 1.5)
        df_derived['CombinedKOPower'] = df_derived['RedRecentKOPower'] + df_derived['BlueRecentKOPower']
    
    # Submission prowess metrics
    if all(col in df.columns for col in ['RedWinsBySubmission', 'RedWins', 'BlueWinsBySubmission', 'BlueWins']):
        df_derived['RedRecentSubProwess'] = np.power(df['RedWinsBySubmission'] / np.maximum(df['RedWins'], 1), 1.5)
        df_derived['BlueRecentSubProwess'] = np.power(df['BlueWinsBySubmission'] / np.maximum(df['BlueWins'], 1), 1.5)
        df_derived['CombinedSubThreat'] = df_derived['RedRecentSubProwess'] + df_derived['BlueRecentSubProwess']
    
    # Age-momentum interactions
    if all(col in df.columns for col in ['RedAge', 'BlueAge', 'RedCurrentWinStreak', 'BlueCurrentWinStreak']):
        df_derived['RedAgeMomentum'] = df['RedCurrentWinStreak'] / np.maximum(df['RedAge'] - 25, 1)
        df_derived['BlueAgeMomentum'] = df['BlueCurrentWinStreak'] / np.maximum(df['BlueAge'] - 25, 1)
    
    # Calculate weight class buckets if weight is available but weight class features aren't
    if 'RedWeightLbs' in df_derived.columns:
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
        
        weight_bucket = categorize_weight(df_derived['RedWeightLbs'].iloc[0])
        
        # Add one-hot encoded weight class features
        for i in range(4):  # 4 weight class buckets
            feature_name = f'WeightClass_{i}'
            df_derived[feature_name] = 1 if weight_bucket == i else 0
    
    return df_derived