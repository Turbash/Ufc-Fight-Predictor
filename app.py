import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from utils.visualization import display_fight_prediction, plot_prediction_confidence
from about import show_about_page

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

def main():
    # Sidebar navigation
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/UFC_Logo.svg/2560px-UFC_Logo.svg.png", width=200)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction Tool", "About"])
    
    if page == "About":
        show_about_page()
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
            # Get feature names that the winner model was trained on
            if hasattr(winner_model, 'get_booster'):
                # For XGBoost models
                winner_features = winner_model.get_booster().feature_names
            else:
                # For other models that don't have direct feature name access
                winner_features = [col for col in fight_data.columns if col not in [
                    'RedDecOdds', 'BlueDecOdds', 'RSubOdds', 'BSubOdds', 'RKOOdds', 'BKOOdds'
                ]]
            
            # Filter fight data to include only features the winner model was trained on
            winner_input_data = fight_data[winner_features]
            
            # Make winner prediction with filtered data
            winner_pred = winner_model.predict(winner_input_data)[0]
            winner_proba = winner_model.predict_proba(winner_input_data)[0]
            
            # For finish model, use the scaler and all available features
            scaled_data = finish_scaler.transform(fight_data)
            finish_pred = finish_model.predict(scaled_data)[0]
            finish_proba = finish_model.predict_proba(scaled_data)[0]
            
            # Get prediction results
            winner_name = red_fighter_name if winner_pred == 0 else blue_fighter_name
            winner_confidence = round(winner_proba[winner_pred] * 100, 1)
            
            finish_types = ['KO/TKO', 'Submission', 'Decision']
            finish_type = finish_types[finish_pred]
            finish_confidence = round(finish_proba[finish_pred] * 100, 1)
            
            # Determine confidence level class
            winner_conf_class = "confidence-high" if winner_confidence >= 70 else "confidence-medium" if winner_confidence >= 55 else "confidence-low"
            finish_conf_class = "confidence-high" if finish_confidence >= 70 else "confidence-medium" if finish_confidence >= 55 else "confidence-low"
            
            # Display predictions
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            st.markdown(f"## Fight Prediction: {red_fighter_name} vs {blue_fighter_name}")
            
            st.markdown(f"### Winner Prediction")
            st.markdown(f"**Predicted Winner:** <span class='{'fighter-red' if winner_pred == 0 else 'fighter-blue'}'>{winner_name}</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span class='{winner_conf_class}'>{winner_confidence}%</span>", unsafe_allow_html=True)
            
            st.markdown(f"### Finish Prediction")
            st.markdown(f"**Predicted Finish:** {finish_type}")
            st.markdown(f"**Confidence:** <span class='{finish_conf_class}'>{finish_confidence}%</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display visualizations
            st.subheader("Prediction Visualizations")
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                labels = [red_fighter_name, blue_fighter_name]
                confidence_vals = [winner_proba[0] * 100, winner_proba[1] * 100]
                colors = ['#E50914', '#3498db']
                bars = ax1.bar(labels, confidence_vals, color=colors)
                ax1.set_ylim(0, 100)
                ax1.set_title('Winner Prediction Confidence (%)')
                ax1.bar_label(bars, fmt='%.1f%%')
                st.pyplot(fig1)
            
            with col_viz2:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                finish_labels = finish_types
                finish_confidence_vals = [finish_proba[i] * 100 for i in range(len(finish_types))]
                finish_colors = ['#e74c3c', '#2ecc71', '#3498db']
                bars = ax2.bar(finish_labels, finish_confidence_vals, color=finish_colors)
                ax2.set_ylim(0, 100)
                ax2.set_title('Finish Type Prediction Confidence (%)')
                ax2.bar_label(bars, fmt='%.1f%%')
                st.pyplot(fig2)

if __name__ == "__main__":
    main()