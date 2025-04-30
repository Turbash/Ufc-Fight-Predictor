import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from utils.visualization import display_fight_prediction, plot_prediction_confidence
from about import show_about_page
from io import StringIO
import sys
sys.path.append('.')
try:
    from extract_features import extract_all_features_from_csv
except ImportError:
    # Define a simplified version if the module is not available
    def extract_all_features_from_csv(df):
        return df

# Set page configuration
st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #E50914;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px #000000;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #f1c40f;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #2c3e50;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .fighter-red {
        color: #E50914;
        font-weight: bold;
    }
    .fighter-blue {
        color: #3498db;
        font-weight: bold;
    }
    .confidence-high {
        color: #2ecc71;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f39c12;
        font-weight: bold;
    .confidence-low {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def load_models():
    """Load the trained models"""
    try:
        winner_model = joblib.load('models/winner_model.pkl')
        finish_model = joblib.load('models/finish_model.pkl')
        finish_scaler = joblib.load('models/finish_scaler.pkl')
        return winner_model, finish_model, finish_scaler
    except:
        st.error("Error loading models. Please make sure the models are trained and saved correctly.")
        return None, None, None

@st.cache_resource
def load_cached_models():
    """Load and cache the prediction models"""
    try:
        winner_model = joblib.load('models/winner_model.pkl')
        finish_model = joblib.load('models/finish_model.pkl')
        finish_scaler = joblib.load('models/finish_scaler.pkl')
        return winner_model, finish_model, finish_scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def main():
    # Sidebar navigation
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/UFC_Logo.svg/2560px-UFC_Logo.svg.png", width=200)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction Tool", "Batch Predictions", "About"])
    
    if page == "About":
        show_about_page()
    elif page == "Batch Predictions":
        show_batch_predictions_page()
    else:
        show_prediction_page()

def show_prediction_page():
    # Header
    st.markdown("<h1 class='main-header'>UFC Fight Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Predict fight outcomes with machine learning</h2>", unsafe_allow_html=True)
    
    # Load models
    winner_model, finish_model, finish_scaler = load_models()
    
    if not winner_model or not finish_model or not finish_scaler:
        st.warning("Please train the models first by running 'python train_models.py'")
        return
    
    # Create two columns for fighter inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 class='fighter-red'>Red Corner Fighter</h3>", unsafe_allow_html=True)
        red_fighter_name = st.text_input("Red Fighter Name", "Fighter 1")
        red_odds = st.number_input("Red Odds (e.g. -150, 250)", value=-150, step=5)
        red_exp_value = st.number_input("Red Expected Value", value=50.0, format="%.1f")
        red_win_streak = st.number_input("Red Win Streak", value=3, min_value=0)
        red_lose_streak = st.number_input("Red Lose Streak", value=0, min_value=0)
        red_height = st.number_input("Red Height (cm)", value=180.0, format="%.1f")
        red_reach = st.number_input("Red Reach (cm)", value=188.0, format="%.1f")
        red_weight = st.number_input("Red Weight (lbs)", value=170.0, format="%.1f")
        red_age = st.number_input("Red Age", value=30, min_value=18)
        red_stance = st.selectbox("Red Stance", ["Orthodox", "Southpaw", "Switch"])
        
        # Fighting stats
        st.subheader("Red Fighter Stats")
        red_sig_str = st.number_input("Red Avg. Sig. Strikes Landed", value=4.5, format="%.2f")
        red_sig_str_pct = st.number_input("Red Sig. Strike Accuracy (%)", value=48.0, format="%.1f") / 100
        red_td_landed = st.number_input("Red Avg. Takedowns Landed", value=1.5, format="%.2f")
        red_td_pct = st.number_input("Red Takedown Accuracy (%)", value=45.0, format="%.1f") / 100
        red_sub_att = st.number_input("Red Avg. Submission Attempts", value=0.8, format="%.2f")
        
        # Win methods
        red_ko_wins = st.number_input("Red KO/TKO Wins", value=5, min_value=0)
        red_sub_wins = st.number_input("Red Submission Wins", value=2, min_value=0)
        red_dec_wins = st.number_input("Red Decision Wins", value=4, min_value=0)
        red_total_wins = st.number_input("Red Total Wins", value=red_ko_wins + red_sub_wins + red_dec_wins, min_value=0)
        red_total_losses = st.number_input("Red Total Losses", value=3, min_value=0)
    
    with col2:
        st.markdown("<h3 class='fighter-blue'>Blue Corner Fighter</h3>", unsafe_allow_html=True)
        blue_fighter_name = st.text_input("Blue Fighter Name", "Fighter 2")
        blue_odds = st.number_input("Blue Odds (e.g. -150, 250)", value=250, step=5)
        blue_exp_value = st.number_input("Blue Expected Value", value=35.0, format="%.1f")
        blue_win_streak = st.number_input("Blue Win Streak", value=2, min_value=0)
        blue_lose_streak = st.number_input("Blue Lose Streak", value=1, min_value=0)
        blue_height = st.number_input("Blue Height (cm)", value=178.0, format="%.1f")
        blue_reach = st.number_input("Blue Reach (cm)", value=185.0, format="%.1f")
        blue_weight = st.number_input("Blue Weight (lbs)", value=170.0, format="%.1f")
        blue_age = st.number_input("Blue Age", value=28, min_value=18)
        blue_stance = st.selectbox("Blue Stance", ["Orthodox", "Southpaw", "Switch"])
        
        # Fighting stats
        st.subheader("Blue Fighter Stats")
        blue_sig_str = st.number_input("Blue Avg. Sig. Strikes Landed", value=3.8, format="%.2f")
        blue_sig_str_pct = st.number_input("Blue Sig. Strike Accuracy (%)", value=45.0, format="%.1f") / 100
        blue_td_landed = st.number_input("Blue Avg. Takedowns Landed", value=2.1, format="%.2f")
        blue_td_pct = st.number_input("Blue Takedown Accuracy (%)", value=50.0, format="%.1f") / 100
        blue_sub_att = st.number_input("Blue Avg. Submission Attempts", value=1.2, format="%.2f")
        
        # Win methods
        blue_ko_wins = st.number_input("Blue KO/TKO Wins", value=4, min_value=0)
        blue_sub_wins = st.number_input("Blue Submission Wins", value=3, min_value=0)
        blue_dec_wins = st.number_input("Blue Decision Wins", value=2, min_value=0)
        blue_total_wins = st.number_input("Blue Total Wins", value=blue_ko_wins + blue_sub_wins + blue_dec_wins, min_value=0)
        blue_total_losses = st.number_input("Blue Total Losses", value=2, min_value=0)
    
    # Fight specific info
    st.subheader("Fight Information")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        title_bout = st.checkbox("Title Bout", value=False)
        red_dec_odds = st.number_input("Red by Decision Odds", value=250, step=5)
        blue_dec_odds = st.number_input("Blue by Decision Odds", value=300, step=5)
    
    with col4:
        num_rounds = st.selectbox("Number of Rounds", [3, 5], index=0)
        red_sub_odds = st.number_input("Red by Submission Odds", value=600, step=5)
        blue_sub_odds = st.number_input("Blue by Submission Odds", value=500, step=5)
    
    with col5:
        weight_class = st.selectbox("Weight Class", 
                                   ["Flyweight", "Bantamweight", "Featherweight", "Lightweight",
                                    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"])
        red_ko_odds = st.number_input("Red by KO/TKO Odds", value=350, step=5)
        blue_ko_odds = st.number_input("Blue by KO/TKO Odds", value=450, step=5)
    
    # Calculate differentials
    win_streak_dif = red_win_streak - blue_win_streak
    lose_streak_dif = red_lose_streak - blue_lose_streak
    height_dif = red_height - blue_height
    reach_dif = red_reach - blue_reach
    age_dif = red_age - blue_age
    sig_str_dif = red_sig_str - blue_sig_str
    avg_sub_att_dif = red_sub_att - blue_sub_att
    avg_td_dif = red_td_landed - blue_td_landed
    
    # Gather data for prediction
    fight_data = pd.DataFrame({
        'RedOdds': [red_odds],
        'BlueOdds': [blue_odds],
        'RedExpectedValue': [red_exp_value],
        'BlueExpectedValue': [blue_exp_value],
        'TitleBout': [1 if title_bout else 0],
        'NumberOfRounds': [num_rounds],
        'RedCurrentLoseStreak': [red_lose_streak],
        'BlueCurrentLoseStreak': [blue_lose_streak],
        'RedCurrentWinStreak': [red_win_streak],
        'BlueCurrentWinStreak': [blue_win_streak],
        'RedAvgSigStrLanded': [red_sig_str],
        'BlueAvgSigStrLanded': [blue_sig_str],
        'RedAvgSigStrPct': [red_sig_str_pct],
        'BlueAvgSigStrPct': [blue_sig_str_pct],
        'RedAvgSubAtt': [red_sub_att],
        'BlueAvgSubAtt': [blue_sub_att],
        'RedAvgTDLanded': [red_td_landed],
        'BlueAvgTDLanded': [blue_td_landed],
        'RedWinsByKO': [red_ko_wins],
        'BlueWinsByKO': [blue_ko_wins],
        'RedWinsBySubmission': [red_sub_wins],
        'BlueWinsBySubmission': [blue_sub_wins],
        'RedWinsByDecisionUnanimous': [red_dec_wins],
        'BlueWinsByDecisionUnanimous': [blue_dec_wins],
        'RedWins': [red_total_wins],
        'BlueWins': [blue_total_wins],
        'RedHeightCms': [red_height],
        'BlueHeightCms': [blue_height],
        'RedReachCms': [red_reach],
        'BlueReachCms': [blue_reach],
        'RedWeightLbs': [red_weight],
        'BlueWeightLbs': [blue_weight],
        'WinStreakDif': [win_streak_dif],
        'LoseStreakDif': [lose_streak_dif],
        'HeightDif': [height_dif],
        'ReachDif': [reach_dif],
        'AgeDif': [age_dif],
        'SigStrDif': [sig_str_dif],
        'AvgSubAttDif': [avg_sub_att_dif],
        'AvgTDDif': [avg_td_dif],
        'RedDecOdds': [red_dec_odds],
        'BlueDecOdds': [blue_dec_odds],
        'RSubOdds': [red_sub_odds],
        'BSubOdds': [blue_sub_odds],
        'RKOOdds': [red_ko_odds],
        'BKOOdds': [blue_ko_odds]
    })
    
    # Predict button
    if st.button("Predict Fight Outcome", key="predict_button"):
        with st.spinner("Analyzing fight data..."):
            # Make predictions
            try:
                # Load the exact features the model was trained on
                winner_features = joblib.load('models/winner_features.pkl')
                winner_scaler = joblib.load('models/winner_scaler.pkl')
                
                # Ensure all required columns exist in fight_data
                for col in winner_features:
                    if col not in fight_data.columns:
                        fight_data[col] = 0  # Default to 0 for missing columns
                
                # Filter and scale the data exactly as it was done during training
                winner_input_data = fight_data[winner_features]
                winner_input_scaled = winner_scaler.transform(winner_input_data)
                
                # Make winner prediction with scaled data
                winner_pred = winner_model.predict(winner_input_scaled)[0]
                winner_proba = winner_model.predict_proba(winner_input_scaled)[0]
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Falling back to unscaled prediction method")
                
                # Fallback to current method if there's a problem
                if hasattr(winner_model, 'get_booster'):
                    winner_features = winner_model.get_booster().feature_names
                else:
                    winner_features = [col for col in fight_data.columns if col not in [
                        'RedDecOdds', 'BlueDecOdds', 'RSubOdds', 'BSubOdds', 'RKOOdds', 'BKOOdds'
                    ]]
                
                winner_input_data = fight_data[winner_features]
                winner_pred = winner_model.predict(winner_input_data)[0]
                winner_proba = winner_model.predict_proba(winner_input_data)[0]
            
            # For finish model, use the scaler and all available features
            scaled_data = finish_scaler.transform(fight_data)
            finish_pred = finish_model.predict(scaled_data)[0]
            finish_proba = finish_model.predict_proba(scaled_data)[0]
            
            # Display visualizations using custom gauge charts instead of bar plots
            st.subheader("Prediction Visualizations")
            
            # Instead of using HTML, use Streamlit's native components
            
            # Get prediction results
            winner_name = red_fighter_name if winner_pred == 0 else blue_fighter_name
            winner_confidence = round(winner_proba[winner_pred] * 100, 1)
            
            finish_types = ['KO/TKO', 'Submission', 'Decision']
            finish_type = finish_types[finish_pred]
            finish_confidence = round(finish_proba[finish_pred] * 100, 1)
            
            # Create fighter cards using Streamlit columns
            st.markdown("### Fighter Win Probability")
            
            fighter_cols = st.columns(2)
            
            # Red fighter card
            with fighter_cols[0]:
                st.markdown(f"##### <span style='color:#E50914;font-size:1.4rem;'>{red_fighter_name}</span>", unsafe_allow_html=True)
                
                # Create a styled container for the card
                red_container = st.container()
                red_container.markdown("""
                <style>
                .red-card {
                    background: linear-gradient(145deg, #1a1e25 0%, #2c3e50 100%);
                    border-radius: 10px;
                    padding: 15px;
                    border-top: 4px solid #E50914;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display content in the styled container
                with red_container:
                    st.metric("Win Probability", f"{round(winner_proba[0] * 100, 1)}%")
                    st.progress(float(winner_proba[0]))
                    if winner_pred == 0:
                        st.success("🏆 Predicted Winner")
            
            # Blue fighter card
            with fighter_cols[1]:
                st.markdown(f"##### <span style='color:#3498db;font-size:1.4rem;'>{blue_fighter_name}</span>", unsafe_allow_html=True)
                
                # Create a styled container for the card
                blue_container = st.container()
                blue_container.markdown("""
                <style>
                .blue-card {
                    background: linear-gradient(145deg, #1a1e25 0%, #2c3e50 100%);
                    border-radius: 10px;
                    padding: 15px;
                    border-top: 4px solid #3498db;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display content in the styled container
                with blue_container:
                    st.metric("Win Probability", f"{round(winner_proba[1] * 100, 1)}%")
                    st.progress(float(winner_proba[1]))
                    if winner_pred == 1:
                        st.success("🏆 Predicted Winner")
            
            # Display finish prediction using Streamlit components
            st.markdown("### Predicted Fight Outcome")
            st.info(f"**{finish_type}** with {finish_confidence}% confidence")
            
            # Finish type probability bars
            finish_cols = st.columns(3)
            
            # KO/TKO probability
            with finish_cols[0]:
                st.markdown("##### KO/TKO")
                ko_prob = round(finish_proba[0] * 100, 1)
                st.metric("Probability", f"{ko_prob}%")
                ko_bar = st.progress(float(finish_proba[0]))
                if finish_pred == 0:
                    st.markdown("✅ **Predicted Method**")
            
            # Submission probability
            with finish_cols[1]:
                st.markdown("##### Submission")
                sub_prob = round(finish_proba[1] * 100, 1)
                st.metric("Probability", f"{sub_prob}%")
                sub_bar = st.progress(float(finish_proba[1]))
                if finish_pred == 1:
                    st.markdown("✅ **Predicted Method**")
            
            # Decision probability
            with finish_cols[2]:
                st.markdown("##### Decision")
                dec_prob = round(finish_proba[2] * 100, 1)
                st.metric("Probability", f"{dec_prob}%")
                dec_bar = st.progress(float(finish_proba[2]))
                if finish_pred == 2:
                    st.markdown("✅ **Predicted Method**")
            
            # Display some additional insights based on the predictions
            st.subheader("Fight Analysis")
            st.write(f"Based on our model, {winner_name} is expected to win this fight via {finish_type.lower()} " +
                    f"with {finish_confidence}% confidence. Key factors contributing to this prediction include " +
                    f"{'favorable odds, fighting experience, and physical advantages' if winner_confidence > 65 else 'slight edge in key statistics'}.")
            
            # Add fighter comparison for key stats
            st.markdown("<h3>Fighter Comparison</h3>", unsafe_allow_html=True)
            comparison_data = {
                "Stat": ["Win Streak", "Sig. Strikes/Min", "Accuracy", "Takedowns/Fight", "KO Wins", "Sub Wins", "Height", "Reach"],
                red_fighter_name: [red_win_streak, red_sig_str, red_sig_str_pct*100, red_td_landed, 
                                  red_ko_wins, red_sub_wins, red_height, red_reach],
                blue_fighter_name: [blue_win_streak, blue_sig_str, blue_sig_str_pct*100, blue_td_landed, 
                                   blue_ko_wins, blue_sub_wins, blue_height, blue_reach]
            }
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create a formatter function to apply custom formatting to specific rows
            def format_df(df):
                # Create a copy to avoid modifying the original
                formatted_df = df.copy()
                
                # Format the accuracy row (index 2) to show as percentage
                for col in formatted_df.columns:
                    if col != "Stat" and len(formatted_df) > 2:
                        formatted_df.loc[2, col] = f"{formatted_df.loc[2, col]:.1f}%"
                
                # Format height and reach rows to show units
                for col in formatted_df.columns:
                    if col != "Stat":
                        if len(formatted_df) > 6:  # Height row
                            formatted_df.loc[6, col] = f"{formatted_df.loc[6, col]} cm"
                        if len(formatted_df) > 7:  # Reach row
                            formatted_df.loc[7, col] = f"{formatted_df.loc[7, col]} cm"
                
                return formatted_df
            
            # Apply formatting and display
            st.dataframe(format_df(comparison_df), use_container_width=True, hide_index=True)

def show_batch_predictions_page():
    # Header
    st.markdown("<h1 class='main-header'>Batch Fight Predictions</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Upload a CSV file to predict multiple fight outcomes</h2>", unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read CSV content
            df = pd.read_csv(uploaded_file)
            
            # Show preview of uploaded data
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Preprocess the DataFrame to ensure all necessary features are present
            processed_df = extract_all_features_from_csv(df)
            
            # Process each fight and generate predictions
            st.subheader("Generating Predictions...")
            progress_bar = st.progress(0)
            results = []
            
            for index, row in processed_df.iterrows():
                # Update progress
                progress = (index + 1) / len(processed_df)
                progress_bar.progress(progress)
                
                # Extract features - now use a dictionary with all columns from the processed DataFrame
                features = row.to_dict()
                
                # Use existing prediction model
                prediction = predict_fight_winner(features)
                
                # Get actual fighter names
                red_fighter = row['RedFighter']
                blue_fighter = row['BlueFighter']
                
                # Determine the predicted winner name
                if prediction['winner'] == 'RedFighter':
                    predicted_winner = red_fighter
                    winner_color = 'red'
                else:
                    predicted_winner = blue_fighter
                    winner_color = 'blue'
                
                # Store results
                results.append({
                    'RedFighter': red_fighter,
                    'BlueFighter': blue_fighter,
                    'WeightClass': row['WeightClass'] if 'WeightClass' in row else 'N/A',
                    'PredictedWinner': predicted_winner,
                    'WinnerColor': winner_color,
                    'Confidence': f"{prediction['confidence'] * 100:.1f}%",
                    'Method': prediction['method'],
                    'MethodConfidence': f"{prediction['method_confidence']}%"
                })
            
            # Display results in a table with colored winners
            st.subheader("Prediction Results")
            results_df = pd.DataFrame(results)
            
            # Apply color formatting using custom HTML
            formatted_df = results_df.copy()
            formatted_df = formatted_df.drop(columns=['WinnerColor']) if 'WinnerColor' in formatted_df.columns else formatted_df
            st.dataframe(formatted_df, use_container_width=True, hide_index=True)
            
            # Option to download results
            csv = results_df.drop(columns=['WinnerColor'] if 'WinnerColor' in results_df.columns else []).to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='fight_predictions.csv',
                mime='text/csv',
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Stack trace:")
            st.exception(e)

def extract_features_for_prediction(row):
    """Extract relevant features from a row of the CSV file for prediction."""
    features = {}
    
    # Map all available columns from CSV to features dict
    for col in row.index:
        if not pd.isna(row[col]):
            features[col] = row[col]
    
    # Calculate differential features like in the main app
    features['WinStreakDif'] = features.get('RedCurrentWinStreak', 0) - features.get('BlueCurrentWinStreak', 0)
    features['LoseStreakDif'] = features.get('RedCurrentLoseStreak', 0) - features.get('BlueCurrentLoseStreak', 0)
    features['HeightDif'] = features.get('RedHeightCms', 0) - features.get('BlueHeightCms', 0)
    features['ReachDif'] = features.get('RedReachCms', 0) - features.get('BlueReachCms', 0)
    features['AgeDif'] = features.get('RedAge', 0) - features.get('BlueAge', 0)
    features['SigStrDif'] = features.get('RedAvgSigStrLanded', 0) - features.get('BlueAvgSigStrLanded', 0)
    features['AvgSubAttDif'] = features.get('RedAvgSubAtt', 0) - features.get('BlueAvgSubAtt', 0)
    features['AvgTDDif'] = features.get('RedAvgTDLanded', 0) - features.get('BlueAvgTDLanded', 0)
    
    # Convert boolean values 
    if 'TitleBout' in features:
        features['TitleBout'] = 1 if features['TitleBout'] else 0
    
    return features

def predict_fight_winner(features):
    """Use the existing prediction model to predict the fight winner."""
    try:
        # Load models only once per session using st.cache_resource
        winner_model, finish_model, finish_scaler = load_cached_models()
        
        # Load features and scaler
        winner_features = joblib.load('models/winner_features.pkl')
        winner_scaler = joblib.load('models/winner_scaler.pkl')
        
        # Create a DataFrame with all necessary columns for the winner model
        winner_df = pd.DataFrame(index=[0])
        for feature in winner_features:
            if feature in features:
                winner_df[feature] = features[feature]
            else:
                winner_df[feature] = 0
        
        # Scale the features
        winner_input_scaled = winner_scaler.transform(winner_df)
        
        # Make winner prediction
        winner_pred = winner_model.predict(winner_input_scaled)[0]
        winner_proba = winner_model.predict_proba(winner_input_scaled)[0]
        
        # For finish prediction, load the exact finish features used during training
        finish_features = joblib.load('models/finish_features.pkl')
        
        # Create DataFrame for finish prediction with exact same columns as training
        finish_df = pd.DataFrame(index=[0])
        for feature in finish_features:
            if feature in features:
                finish_df[feature] = features[feature]
            else:
                finish_df[feature] = 0
        
        # Use the same scaling approach as in training
        scaled_finish_data = finish_scaler.transform(finish_df)
        
        # Make predictions with scaled data
        finish_pred = finish_model.predict(scaled_finish_data)[0]
        finish_proba = finish_model.predict_proba(scaled_finish_data)[0]
        
        # Map finish prediction to human-readable string
        finish_types = ['KO/TKO', 'Submission', 'Decision']
        finish_type = finish_types[finish_pred]
        finish_confidence = round(finish_proba[finish_pred] * 100, 1)
        
        return {
            'winner': 'RedFighter' if winner_pred == 0 else 'BlueFighter',
            'confidence': round(winner_proba[winner_pred], 2),
            'method': finish_type,
            'method_confidence': finish_confidence
        }
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        # If there's an error loading features or predictions, log it and use a default
        st.exception(e)
        return {
            'winner': 'Error',
            'confidence': 0.5,
            'method': 'Error',
            'method_confidence': 0,
            'error': str(e)
        }

if __name__ == "__main__":
    main()