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
    
    # Copy over basic fight information
    processed_df['RedFighter'] = df['RedFighter']
    processed_df['BlueFighter'] = df['BlueFighter']
    processed_df['WeightClass'] = df['WeightClass']
    
    # Process odds and values
    for col in ['RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue']:
        if col in df.columns:
            processed_df[col] = df[col].astype(float)
    
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
            processed_df[col] = df[col].fillna(0).astype(float)
    
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
    
    for dif_name, (red_col, blue_col) in dif_calculations.items():
        if red_col in processed_df.columns and blue_col in processed_df.columns:
            processed_df[dif_name] = processed_df[red_col] - processed_df[blue_col]
    
    return processed_df

def save_model_features():
    """
    Extract and save the feature sets used by the models for consistent prediction.
    """
    try:
        # Load models
        winner_model = joblib.load('models/winner_model.pkl')
        finish_model = joblib.load('models/finish_model.pkl')
        
        # Get features from winner model
        if hasattr(winner_model, 'feature_names_'):
            winner_features = winner_model.feature_names_
        elif hasattr(winner_model, 'get_booster') and hasattr(winner_model.get_booster(), 'feature_names'):
            winner_features = winner_model.get_booster().feature_names
        else:
            # If can't extract directly, use a default set
            winner_features = ['RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue',
                              'RedAge', 'BlueAge', 'RedCurrentWinStreak', 'BlueCurrentWinStreak',
                              'RedAvgSigStrLanded', 'BlueAvgSigStrLanded', 'RedAvgTDLanded', 
                              'BlueAvgTDLanded', 'WinStreakDif', 'AgeDif', 'SigStrDif']
        
        # Get features from finish model
        if hasattr(finish_model, 'feature_names_'):
            finish_features = finish_model.feature_names_
        elif hasattr(finish_model, 'get_booster') and hasattr(finish_model.get_booster(), 'feature_names'):
            finish_features = finish_model.get_booster().feature_names
        else:
            # If can't extract directly, use a default set
            finish_features = ['RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue',
                               'NumberOfRounds', 'RedWinsByKO', 'BlueWinsByKO', 
                               'RedWinsBySubmission', 'BlueWinsBySubmission']
        
        # Save the feature lists
        joblib.dump(winner_features, 'models/winner_features.pkl')
        joblib.dump(finish_features, 'models/finish_features.pkl')
        
        print("Model features saved successfully")
        return True
    except Exception as e:
        print(f"Error saving model features: {str(e)}")
        return False

if __name__ == "__main__":
    # Run this to save feature lists for both models
    save_model_features()
