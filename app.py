import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from utils.visualization import display_fight_prediction, plot_prediction_confidence
from utils.preprocess import preprocess_winner_input, preprocess_finish_input
from about import show_about_page
from io import StringIO
import sys
import traceback

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

# Custom CSS for improved UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #d20a0a;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ffc107;
        margin-bottom: 1rem;
        text-align: center;
    }
    .prediction-box {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #333;
    }
    .fighter-red {
        color: #d20a0a;
        font-weight: bold;
    }
    .fighter-blue {
        color: #0a7ad2;
        font-weight: bold;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    /* Modern card styling */
    .modern-card {
        background: linear-gradient(145deg, #1a1e25 0%, #2c3e50 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        border-left: 5px solid #333;
    }
    .red-card {
        border-left: 5px solid #d20a0a;
    }
    .blue-card {
        border-left: 5px solid #0a7ad2;
    }
    .gold-card {
        border-left: 5px solid #ffc107;
    }
    /* Improved button styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
        color: white;
        font-size: 20px;
        font-weight: bold;
        padding: 12px 30px;
        border-radius: 50px;
        border: none;
        display: block;
        margin: 30px auto;
        width: 90%;
        max-width: 400px;
        transition: all 0.3s ease;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
        box-shadow: 0 4px 15px rgba(255, 8, 68, 0.4);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #ff0844 0%, #ff3c6f 100%);
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 7px 25px rgba(255, 8, 68, 0.5);
    }
    div.stButton > button:first-child:active {
        transform: translateY(1px);
    }
    /* Progress bar styling */
    div.stProgress > div > div > div {
        background-image: linear-gradient(to right, #0a7ad2, #d20a0a);
    }
    /* Custom metric styling */
    div.css-12w0qpk.e1tzin5v2 {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #333;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .viewerBadge_container__1QSob {display: none;}
</style>
""", unsafe_allow_html=True)

def find_models():
    """Search for model files in various potential locations"""
    possible_model_dirs = [
        'models',
        'models/stacked_calibrated',
        'models/stacked_uncalibrated',
        'models/calibrated',
        'models/uncalibrated',
        '.'
    ]
    
    # Try to find where models are stored
    for directory in possible_model_dirs:
        winner_path = os.path.join(directory, 'winner_model.pkl')
        finish_path = os.path.join(directory, 'finish_model.pkl')
        
        if os.path.exists(winner_path) and os.path.exists(finish_path):
            print(f"Found models in directory: {directory}")
            return directory
    
    return None

def load_models():
    """Load the trained models with improved path detection"""
    try:
        # Find where the models are stored
        model_dir = find_models()
        
        if not model_dir:
            st.error("Model files not found. Please ensure you have run train_models.py successfully.")
            st.info("Looking for models in: models/, models/stacked_calibrated/, models/calibrated/, etc.")
            return None, None, None, None, None
        
        # Construct paths using the found directory
        winner_model_path = os.path.join(model_dir, 'winner_model.pkl')
        finish_model_path = os.path.join(model_dir, 'finish_model.pkl')
        winner_scaler_path = os.path.join(model_dir, 'winner_scaler.pkl')
        finish_scaler_path = os.path.join(model_dir, 'finish_scaler.pkl')
        winner_features_path = os.path.join(model_dir, 'winner_features.pkl')
        finish_features_path = os.path.join(model_dir, 'finish_features.pkl')
        
        st.success(f"Loading models from: {model_dir}")
            
        # Load both models
        winner_model = joblib.load(winner_model_path)
        finish_model = joblib.load(finish_model_path)
        
        # Load feature lists to know which features are actually used
        try:
            winner_features = joblib.load(winner_features_path)
            finish_features = joblib.load(finish_features_path)
        except Exception as e:
            st.warning(f"Couldn't load feature lists: {str(e)}")
            winner_features = None
            finish_features = None
        
        # Load scalers for proper data preprocessing
        try:
            winner_scaler = joblib.load(winner_scaler_path)
            finish_scaler = joblib.load(finish_scaler_path)
        except Exception as e:
            st.warning(f"Scalers not found: {str(e)}. Model predictions may be less accurate.")
            winner_scaler = None
            finish_scaler = None
        
        return winner_model, finish_model, finish_scaler, winner_features, finish_features
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.exception(e)
        return None, None, None, None, None

@st.cache_resource
def load_cached_models():
    """Load and cache the prediction models"""
    return load_models()

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
    
    # Load models and feature lists
    winner_model, finish_model, finish_scaler, winner_features, finish_features = load_cached_models()
    
    if not winner_model or not finish_model:
        st.warning("Please train the models first by running 'python train_models.py'")
        return
    
    # Create two columns for fighter inputs with improved UI
    col1, col2 = st.columns(2)
    
    # Dictionary to store all input values
    fight_data_dict = {}
    
    with col1:
        st.markdown("<h3 style='color:#d20a0a; text-align:center;'>Red Corner Fighter</h3>", unsafe_allow_html=True)
        
        # Add a card-like container for the fighter
        with st.container():
            st.markdown("<div class='modern-card red-card'>", unsafe_allow_html=True)
            
            red_fighter_name = st.text_input("Red Fighter Name", "Fighter 1")
            
            # Physical attributes section
            st.subheader("Physical Attributes")
            col1a, col1b = st.columns(2)
            
            with col1a:
                fight_data_dict['RedHeightCms'] = st.number_input("Height (cm)", value=180.0, format="%.1f", key="red_height")
                fight_data_dict['RedWeightLbs'] = st.number_input("Weight (lbs)", value=155.0, format="%.1f", key="red_weight")
            
            with col1b:
                fight_data_dict['RedReachCms'] = st.number_input("Reach (cm)", value=183.0, format="%.1f", key="red_reach")
                fight_data_dict['RedAge'] = st.number_input("Age", value=29, min_value=18, max_value=50, key="red_age")
            
            # Basic stats section
            st.subheader("Fight Record")
            col1c, col1d = st.columns(2)
            
            with col1c:
                fight_data_dict['RedWins'] = st.number_input("Wins", value=10, min_value=0, key="red_wins")
                fight_data_dict['RedCurrentWinStreak'] = st.number_input("Win Streak", value=3, min_value=0, key="red_winstreak")
            
            with col1d:
                fight_data_dict['RedLosses'] = st.number_input("Losses", value=2, min_value=0, key="red_losses")
                fight_data_dict['RedCurrentLoseStreak'] = st.number_input("Lose Streak", value=0, min_value=0, key="red_losestreak")
            
            # Win methods section
            st.subheader("Win Methods")
            col1e, col1f, col1g = st.columns(3)
            
            with col1e:
                fight_data_dict['RedWinsByKO'] = st.number_input("KO/TKO", value=5, min_value=0, key="red_ko_wins")
            
            with col1f:
                fight_data_dict['RedWinsBySubmission'] = st.number_input("Submission", value=2, min_value=0, key="red_sub_wins")
            
            with col1g:
                fight_data_dict['RedWinsByDecisionUnanimous'] = st.number_input("Decision", value=3, min_value=0, key="red_dec_wins")
            
            # Performance stats section - in an expander for cleaner UI
            with st.expander("Performance Stats", expanded=False):
                col1h, col1i = st.columns(2)
                
                with col1h:
                    fight_data_dict['RedAvgSigStrLanded'] = st.number_input("Sig. Strikes/Min", value=4.5, format="%.2f", key="red_strikes")
                    fight_data_dict['RedAvgTDLanded'] = st.number_input("Takedowns/15min", value=1.5, format="%.2f", key="red_td")
                
                with col1i:
                    fight_data_dict['RedAvgSigStrPct'] = st.number_input("Strike Accuracy %", value=48.0, format="%.1f", key="red_acc") / 100
                    fight_data_dict['RedAvgSubAtt'] = st.number_input("Submission Att/15min", value=0.8, format="%.2f", key="red_subatt")
            
            # Odds
            st.subheader("Betting Odds")
            fight_data_dict['RedOdds'] = st.number_input("Moneyline Odds (e.g. -150, 250)", value=-150, step=5, key="red_odds")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3 style='color:#0a7ad2; text-align:center;'>Blue Corner Fighter</h3>", unsafe_allow_html=True)
        
        # Add a card-like container for the fighter
        with st.container():
            st.markdown("<div class='modern-card blue-card'>", unsafe_allow_html=True)
            
            blue_fighter_name = st.text_input("Blue Fighter Name", "Fighter 2")
            
            # Physical attributes section
            st.subheader("Physical Attributes")
            col2a, col2b = st.columns(2)
            
            with col2a:
                fight_data_dict['BlueHeightCms'] = st.number_input("Height (cm)", value=178.0, format="%.1f", key="blue_height")
                fight_data_dict['BlueWeightLbs'] = st.number_input("Weight (lbs)", value=155.0, format="%.1f", key="blue_weight")
            
            with col2b:
                fight_data_dict['BlueReachCms'] = st.number_input("Reach (cm)", value=181.0, format="%.1f", key="blue_reach")
                fight_data_dict['BlueAge'] = st.number_input("Age", value=31, min_value=18, max_value=50, key="blue_age")
            
            # Basic stats section
            st.subheader("Fight Record")
            col2c, col2d = st.columns(2)
            
            with col2c:
                fight_data_dict['BlueWins'] = st.number_input("Wins", value=8, min_value=0, key="blue_wins")
                fight_data_dict['BlueCurrentWinStreak'] = st.number_input("Win Streak", value=2, min_value=0, key="blue_winstreak")
            
            with col2d:
                fight_data_dict['BlueLosses'] = st.number_input("Losses", value=3, min_value=0, key="blue_losses")
                fight_data_dict['BlueCurrentLoseStreak'] = st.number_input("Lose Streak", value=1, min_value=0, key="blue_losestreak")
            
            # Win methods section
            st.subheader("Win Methods")
            col2e, col2f, col2g = st.columns(3)
            
            with col2e:
                fight_data_dict['BlueWinsByKO'] = st.number_input("KO/TKO", value=4, min_value=0, key="blue_ko_wins")
            
            with col2f:
                fight_data_dict['BlueWinsBySubmission'] = st.number_input("Submission", value=2, min_value=0, key="blue_sub_wins")
            
            with col2g:
                fight_data_dict['BlueWinsByDecisionUnanimous'] = st.number_input("Decision", value=2, min_value=0, key="blue_dec_wins")
            
            # Performance stats section - in an expander for cleaner UI
            with st.expander("Performance Stats", expanded=False):
                col2h, col2i = st.columns(2)
                
                with col2h:
                    fight_data_dict['BlueAvgSigStrLanded'] = st.number_input("Sig. Strikes/Min", value=3.8, format="%.2f", key="blue_strikes")
                    fight_data_dict['BlueAvgTDLanded'] = st.number_input("Takedowns/15min", value=2.1, format="%.2f", key="blue_td")
                
                with col2i:
                    fight_data_dict['BlueAvgSigStrPct'] = st.number_input("Strike Accuracy %", value=45.0, format="%.1f", key="blue_acc") / 100
                    fight_data_dict['BlueAvgSubAtt'] = st.number_input("Submission Att/15min", value=1.2, format="%.2f", key="blue_subatt")
            
            # Odds
            st.subheader("Betting Odds")
            fight_data_dict['BlueOdds'] = st.number_input("Moneyline Odds (e.g. -150, 250)", value=250, step=5, key="blue_odds")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Fight information section with improved styling
    st.markdown("<h3 style='color:#ffc107; text-align:center; margin-top: 20px;'>Fight Details</h3>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='modern-card gold-card'>", unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            fight_data_dict['NumberOfRounds'] = st.selectbox("Number of Rounds", [3, 5], index=0)
            fight_data_dict['TitleBout'] = 1 if st.checkbox("Title Bout", value=False) else 0
        
        with col4:
            # WeightClass is not needed for input as we'll calculate the weight class from weight
            weight_class = st.selectbox("Weight Class", 
                                ["Flyweight", "Bantamweight", "Featherweight", "Lightweight",
                                 "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"], 
                                index=3)  # Default to Lightweight
            # We don't need to store weight_class in fight_data_dict as we'll calculate it
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create DataFrame from the collected data
    fight_data = pd.DataFrame([fight_data_dict])
    
    # Predict button with improved styling
    if st.button("Predict Fight Outcome", key="predict_button"):
        with st.spinner("Analyzing fight data..."):
            # Make predictions
            try:
                # Use the preprocessing utilities to properly scale the data
                winner_input_scaled = preprocess_winner_input(fight_data)
                
                # Make winner prediction using properly scaled data
                winner_pred = winner_model.predict(winner_input_scaled)[0]
                winner_proba = winner_model.predict_proba(winner_input_scaled)[0]
                
                # For finish model, use preprocessing utility
                finish_input_scaled = preprocess_finish_input(fight_data)
                
                # Make finish prediction with scaled data
                finish_pred = finish_model.predict(finish_input_scaled)[0]
                finish_proba = finish_model.predict_proba(finish_input_scaled)[0]
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Falling back to alternative prediction method")
                
                # Fallback to simple approach with manual scaling
                try:
                    # Load the scaler if available
                    winner_scaler = joblib.load('models/winner_scaler.pkl')
                    winner_features = joblib.load('models/winner_features.pkl')
                    
                    # Ensure all required columns exist in fight_data
                    for col in winner_features:
                        if col not in fight_data.columns:
                            fight_data[col] = 0
                    
                    winner_input_data = fight_data[winner_features]
                    winner_input_scaled = winner_scaler.transform(winner_input_data)
                    
                    winner_pred = winner_model.predict(winner_input_scaled)[0]
                    winner_proba = winner_model.predict_proba(winner_input_scaled)[0]
                except:
                    # Last resort - try unscaled prediction
                    if hasattr(winner_model, 'get_booster'):
                        winner_features = winner_model.get_booster().feature_names
                    else:
                        winner_features = [col for col in fight_data.columns if col not in [
                            'RedDecOdds', 'BlueDecOdds', 'RSubOdds', 'BSubOdds', 'RKOOdds', 'BKOOdds'
                        ]]
                    
                    winner_input_data = fight_data[winner_features]
                    winner_pred = winner_model.predict(winner_input_data)[0]
                    winner_proba = winner_model.predict_proba(winner_input_data)[0]
            
            # For finish model, use preprocessing utility
            try:
                finish_input_scaled = preprocess_finish_input(fight_data)
                
                # Make finish prediction with scaled data
                finish_pred = finish_model.predict(finish_input_scaled)[0]
                finish_proba = finish_model.predict_proba(finish_input_scaled)[0]
            except Exception as e:
                st.error(f"Error making finish prediction: {str(e)}")
                # Fallback to manual scaling
                try:
                    finish_features = joblib.load('models/finish_features.pkl')
                    finish_scaler = joblib.load('models/finish_scaler.pkl')
                    
                    finish_input_data = pd.DataFrame(index=[0])
                    for feature in finish_features:
                        if feature in fight_data.columns:
                            finish_input_data[feature] = fight_data[feature].values[0]
                        else:
                            finish_input_data[feature] = 0
                            
                    finish_input_scaled = finish_scaler.transform(finish_input_data)
                    finish_pred = finish_model.predict(finish_input_scaled)[0]
                    finish_proba = finish_model.predict_proba(finish_input_scaled)[0]
                except:
                    # Last resort fallback
                    finish_features = joblib.load('models/finish_features.pkl')
                    finish_input_data = fight_data[finish_features]
                    finish_pred = finish_model.predict(finish_input_data)[0]
                    finish_proba = finish_model.predict_proba(finish_input_data)[0]
            
            # Get prediction results
            winner_name = red_fighter_name if winner_pred == 0 else blue_fighter_name
            winner_confidence = round(winner_proba[winner_pred] * 100, 1)
            
            # Updated finish types for binary classification
            finish_types = ['Finish', 'Decision']
            finish_type = finish_types[finish_pred]
            finish_confidence = round(finish_proba[finish_pred] * 100, 1)
            
            # Display results in a modern card
            st.markdown("<div class='modern-card prediction-box'>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align:center; color:#ffc107;'>Fight Prediction Results</h2>", unsafe_allow_html=True)
            
            # Main prediction result - Create a VS style card using Streamlit components instead of raw HTML
            st.markdown(f"<h3 style='text-align:center; color:#ffc107;'>Fight Matchup</h3>", unsafe_allow_html=True)

            # Use Streamlit columns for the VS display instead of HTML flex
            col_red, col_vs, col_blue = st.columns([45, 10, 45])

            with col_red:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; 
                     background-color: rgba(210,10,10,{0.7 if winner_pred == 0 else 0.3}); 
                     border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); position: relative;">
                    <h3 style="color:#ffffff; margin: 0;">Red Corner</h3>
                    <h1 style="color:#ffffff; font-size:2rem; margin: 10px 0;">{red_fighter_name}</h1>
                    <p style="color:#ffc107; font-size:1.2rem; margin: 5px 0;">{round(winner_proba[0] * 100, 1)}% Win Chance</p>
                    {f'<span style="position: absolute; top: 10px; left: 10px; background-color: #d20a0a; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold;">WINNER</span>' if winner_pred == 0 else ''}
                </div>
                """, unsafe_allow_html=True)

            with col_vs:
                st.markdown(f"""
                <div style="text-align: center; height: 100%; display: flex; align-items: center; justify-content: center;">
                    <div style="background-color: #ffc107; color: #000; border-radius: 50%; width: 40px; height: 40px; line-height: 40px; font-weight: bold; box-shadow: 0 0 15px rgba(255,193,7,0.7);">VS</div>
                </div>
                """, unsafe_allow_html=True)

            with col_blue:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; 
                     background-color: rgba(10,122,210,{0.7 if winner_pred == 1 else 0.3}); 
                     border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); position: relative;">
                    <h3 style="color:#ffffff; margin: 0;">Blue Corner</h3>
                    <h1 style="color:#ffffff; font-size:2rem; margin: 10px 0;">{blue_fighter_name}</h1>
                    <p style="color:#ffc107; font-size:1.2rem; margin: 5px 0;">{round(winner_proba[1] * 100, 1)}% Win Chance</p>
                    {f'<span style="position: absolute; top: 10px; right: 10px; background-color: #0a7ad2; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold;">WINNER</span>' if winner_pred == 1 else ''}
                </div>
                """, unsafe_allow_html=True)

            # Display winner announcement separately
            st.markdown(f"""
            <div style="text-align:center; padding:20px; margin:20px 0; background: linear-gradient(145deg, #1a1e25 0%, #2c3e50 100%); border-radius: 10px;">
                <h2 style="color:#ffc107; font-size:1.8rem; margin-bottom:10px;">Predicted Winner</h2>
                <h1 style="color:{'#d20a0a' if winner_pred == 0 else '#0a7ad2'}; font-size:3rem; 
                     text-shadow: 0px 0px 10px rgba({'210,10,10' if winner_pred == 0 else '10,122,210'},0.7);
                     margin:5px 0;">
                    {winner_name}
                </h1>
                <div style="display: inline-block; background-color: #1e1e1e; padding: 8px 20px; border-radius: 30px; margin-top:10px;">
                    <p style="color:#ffffff; font-size:1.3rem; margin:0;">
                        <span style="color:#ffc107; font-weight:bold;">{winner_confidence}%</span> Confidence
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display fighter comparison using modern Plotly charts
            st.markdown("<h3 style='text-align:center; margin-top:30px;'>Fighter Comparison</h3>", unsafe_allow_html=True)
            
            # Winner probability visualization - replace gauge with a more attractive design
            st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
            st.subheader("Win Probability")
            
            # Create a better win probability visualization
            fig = go.Figure()
            
            # Add red fighter probability segment
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=winner_proba[0] * 100,
                domain={'x': [0, 0.45], 'y': [0, 1]},
                title={'text': red_fighter_name, 'font': {'color': '#d20a0a', 'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'visible': False},
                    'bar': {'color': "#d20a0a"},
                    'bgcolor': "rgba(210, 10, 10, 0.1)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 100], 'color': "rgba(210, 10, 10, 0.1)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.8,
                        'value': 50
                    }
                },
                number={'suffix': "%", 'font': {'size': 24, 'color': "#d20a0a", 'family': "Arial Black"}}
            ))
            
            # Add blue fighter probability segment
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=winner_proba[1] * 100,
                domain={'x': [0.55, 1], 'y': [0, 1]},
                title={'text': blue_fighter_name, 'font': {'color': '#0a7ad2', 'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'visible': False},
                    'bar': {'color': "#0a7ad2"},
                    'bgcolor': "rgba(10, 122, 210, 0.1)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 100], 'color': "rgba(10, 122, 210, 0.1)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.8,
                        'value': 50
                    }
                },
                number={'suffix': "%", 'font': {'size': 24, 'color': "#0a7ad2", 'family': "Arial Black"}}
            ))
            
            # Add VS in the middle
            fig.add_annotation(
                x=0.5, y=0.5,
                text="VS",
                font=dict(size=24, color="#ffc107", family="Arial Black"),
                showarrow=False,
                xref="paper",
                yref="paper"
            )
            
            # Update layout
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=70, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                grid={'rows': 1, 'columns': 2},
                annotations=[
                    dict(
                        x=0.5, y=1,
                        xref="paper",
                        yref="paper",
                        text="Win Probability Comparison",
                        font=dict(size=20, color="white"),
                        showarrow=False
                    )
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Finish type visualization - create a modern donut chart
            fig2 = go.Figure()
            
            # Add a more attractive donut chart for finish type
            fig2.add_trace(go.Pie(
                values=[finish_proba[0] * 100, finish_proba[1] * 100],
                labels=['Finish', 'Decision'],
                hole=0.7,
                textinfo='label+percent',
                marker=dict(
                    colors=['#d20a0a', '#0a7ad2'],
                    line=dict(color='#ffffff', width=2)
                ),
                textfont=dict(size=14, family="Arial"),
                insidetextorientation='radial'
            ))

            # Add glowing effect with shadow
            fig2.update_layout(
                title={
                    'text': "Finish Type Prediction",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 24, 'color': '#ffffff', 'family': "Arial"}
                },
                height=380,  # Increase height to accommodate legend
                margin=dict(l=20, r=20, t=70, b=70),  # Increase bottom margin
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="white", family="Arial"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.25,  # Move legend further down
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="white",
                    borderwidth=1
                ),
                annotations=[
                    # Add finish type in the center
                    dict(
                        x=0.5, y=0.5,
                        text=finish_type,
                        font=dict(size=28, color="#ffc107", family="Arial Black"),
                        showarrow=False
                    ),
                    # Add confidence percentage
                    dict(
                        x=0.5, y=0.4,
                        text=f"{finish_confidence}%",
                        font=dict(size=18, color="#ffffff", family="Arial"),
                        showarrow=False
                    )
                ]
            )

            # Add finishing move icons or graphics based on the prediction
            if finish_type == "Finish":
                # Add KO icon annotation
                fig2.add_annotation(
                    x=0.5, y=0.65,
                    text="💥",  # KO/TKO symbol
                    font=dict(size=30),
                    showarrow=False
                )
            else:
                # Add decision icon annotation
                fig2.add_annotation(
                    x=0.5, y=0.65,
                    text="🏆",  # Decision symbol
                    font=dict(size=30),
                    showarrow=False
                )

            st.plotly_chart(fig2, use_container_width=True)

def show_batch_predictions_page():
    # Header
    st.markdown("<h1 class='main-header'>Batch Fight Predictions</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Upload a CSV file to predict multiple fight outcomes</h2>", unsafe_allow_html=True)
    
    # Display instructions for CSV format
    with st.expander("CSV File Requirements", expanded=True):
        st.markdown("""
        ### Required CSV Format
        
        Your CSV file should contain the following columns for accurate predictions:
        
        **Basic Information (Required)**:
        - `RedFighter`, `BlueFighter`: Fighter names
        - `RedOdds`, `BlueOdds`: Betting odds (e.g., -150, 250)
        - `NumberOfRounds`: Number of rounds (typically 3 or 5)
        - `TitleBout`: Whether it's a title bout (0 or 1)
        
        **Fighter Records**:
        - `RedWins`, `BlueWins`: Total wins
        - `RedWinsByKO`, `BlueWinsByKO`: Wins by KO/TKO
        - `RedWinsBySubmission`, `BlueWinsBySubmission`: Wins by submission
        - `RedCurrentWinStreak`, `BlueCurrentWinStreak`: Current win streak
        
        **Fighter Stats**:
        - `RedAvgSigStrLanded`, `BlueAvgSigStrLanded`: Significant strikes landed per minute
        - `RedAvgSigStrPct`, `BlueAvgSigStrPct`: Significant strike accuracy (0-1)
        - `RedAvgTDLanded`, `BlueAvgTDLanded`: Takedowns landed per 15 minutes
        
        **Physical Attributes**:
        - `RedHeightCms`, `BlueHeightCms`: Height in centimeters
        - `RedReachCms`, `BlueReachCms`: Reach in centimeters
        - `RedWeightLbs`, `BlueWeightLbs`: Weight in pounds
        
        [Download Sample CSV Template](https://github.com/Turbash/Ufc-Fight-Predictor/raw/main/assets/sample_batch_prediction.csv)
        """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read CSV content
            df = pd.read_csv(uploaded_file)
            
            # Show preview of uploaded data
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Check for required columns
            required_columns = ['RedFighter', 'BlueFighter', 'RedOdds', 'BlueOdds']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.stop()
            
            # Preprocess the DataFrame to ensure all necessary features are present
            try:
                # Import feature engineering function
                from utils.preprocess import calculate_derived_features
                
                # Calculate derived features
                processed_df = calculate_derived_features(df)
                
                # If extract_all_features_from_csv is available, use it for additional processing
                try:
                    from extract_features import extract_all_features_from_csv
                    processed_df = extract_all_features_from_csv(processed_df)
                    st.success("Applied advanced feature extraction")
                except ImportError:
                    st.info("Using basic feature processing only")
                
            except Exception as e:
                st.error(f"Error in feature extraction: {str(e)}")
                st.error(traceback.format_exc())
                processed_df = df  # Fallback to original dataframe
            
            # Load models
            winner_model, finish_model, finish_scaler, winner_features, finish_features = load_cached_models()
            if not winner_model or not finish_model:
                st.error("Failed to load prediction models")
                return
            
            # Process each fight and generate predictions
            st.subheader("Generating Predictions...")
            
            # Create progress tracking
            progress_text = "Processing fights..."
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for index, row in processed_df.iterrows():
                # Update progress
                progress = (index + 1) / len(processed_df)
                progress_bar.progress(progress)
                status_text.text(f"Processing fight {index+1} of {len(processed_df)}: {row.get('RedFighter', 'Red')} vs {row.get('BlueFighter', 'Blue')}")
                
                # Create a DataFrame with the current fight's data
                fight_data = pd.DataFrame([row.to_dict()])
                
                # Make prediction using utilities that handle scaling
                try:
                    # Process data for winner prediction
                    from utils.preprocess import preprocess_winner_input, preprocess_finish_input
                    
                    # Winner prediction
                    winner_input = preprocess_winner_input(fight_data)
                    winner_pred = winner_model.predict(winner_input)[0]
                    winner_proba = winner_model.predict_proba(winner_input)[0]
                    
                    # Apply temperature scaling for more reasonable probabilities
                    if np.max(winner_proba) > 0.95:
                        temperature = 2.0
                        logits = np.log(winner_proba / (1 - winner_proba + 1e-10))
                        winner_proba = 1/(1 + np.exp(-logits/temperature))
                        winner_proba = winner_proba / np.sum(winner_proba)  # Normalize
                    
                    # Finish prediction
                    finish_input = preprocess_finish_input(fight_data)
                    finish_pred = finish_model.predict(finish_input)[0]
                    finish_proba = finish_model.predict_proba(finish_input)[0]
                    
                    # Apply temperature scaling for finish probabilities
                    if np.max(finish_proba) > 0.95:
                        temperature = 2.0
                        logits = np.log(finish_proba / (1 - finish_proba + 1e-10))
                        finish_proba = 1/(1 + np.exp(-logits/temperature))
                        finish_proba = finish_proba / np.sum(finish_proba)  # Normalize
                    
                    # Get actual fighter names
                    red_fighter = row['RedFighter'] if 'RedFighter' in row else f'Red Fighter {index+1}'
                    blue_fighter = row['BlueFighter'] if 'BlueFighter' in row else f'Blue Fighter {index+1}'
                    
                    # Determine the predicted winner name
                    predicted_winner = red_fighter if winner_pred == 0 else blue_fighter
                    winner_color = 'red' if winner_pred == 0 else 'blue'
                    
                    # Get finish type - updated for binary classification
                    finish_types = ['Finish', 'Decision']
                    finish_type = finish_types[finish_pred]
                    finish_confidence = round(finish_proba[finish_pred] * 100, 1)
                    
                    # Store results
                    results.append({
                        'RedFighter': red_fighter,
                        'BlueFighter': blue_fighter,
                        'WeightClass': row.get('WeightClass', 'N/A'),
                        'PredictedWinner': predicted_winner,
                        'WinnerColor': winner_color,
                        'WinnerConfidence': round(winner_proba[winner_pred] * 100, 1),
                        'Method': finish_type,
                        'MethodConfidence': finish_confidence,
                        'RedOdds': row.get('RedOdds', 'N/A'),
                        'BlueOdds': row.get('BlueOdds', 'N/A')
                    })
                except Exception as e:
                    st.error(f"Error processing fight {index+1}: {str(e)}")
                    st.error(traceback.format_exc())
                    # Add error entry
                    results.append({
                        'RedFighter': row.get('RedFighter', f'Fighter {index*2+1}'),
                        'BlueFighter': row.get('BlueFighter', f'Fighter {index*2+2}'),
                        'WeightClass': row.get('WeightClass', 'N/A'),
                        'PredictedWinner': 'Error',
                        'WinnerColor': 'N/A',
                        'WinnerConfidence': 0.0,
                        'Method': 'Error',
                        'MethodConfidence': 0.0,
                        'RedOdds': row.get('RedOdds', 'N/A'),
                        'BlueOdds': row.get('BlueOdds', 'N/A')
                    })
            
            # Clear progress indicators when complete
            progress_bar.empty()
            status_text.empty()
            
            # Display results in a table with colored winners
            st.subheader("Prediction Results")
            results_df = pd.DataFrame(results)
            
            # Create a displayable version without the color column
            display_columns = [col for col in results_df.columns if col != 'WinnerColor']
            st.dataframe(results_df[display_columns], use_container_width=True, hide_index=True)
            
            # Show an analysis summary
            with st.expander("Batch Prediction Analysis", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Calculate how many red/blue corner wins were predicted
                    red_wins = sum(1 for result in results if result['WinnerColor'] == 'red' and result['PredictedWinner'] != 'Error')
                    blue_wins = sum(1 for result in results if result['WinnerColor'] == 'blue' and result['PredictedWinner'] != 'Error')
                    total_valid = red_wins + blue_wins
                    
                    st.metric("Red Corner Wins", f"{red_wins} ({round(red_wins/total_valid*100 if total_valid else 0)}%)")
                    st.metric("Blue Corner Wins", f"{blue_wins} ({round(blue_wins/total_valid*100 if total_valid else 0)}%)")
                
                with col2:
                    # Calculate finish vs decision predictions
                    finishes = sum(1 for result in results if result['Method'] == 'Finish')
                    decisions = sum(1 for result in results if result['Method'] == 'Decision')
                    total_valid = finishes + decisions
                    
                    st.metric("Predicted Finishes", f"{finishes} ({round(finishes/total_valid*100 if total_valid else 0)}%)")
                    st.metric("Predicted Decisions", f"{decisions} ({round(decisions/total_valid*100 if total_valid else 0)}%)")
                
                with col3:
                    # Calculate average confidences
                    avg_winner_conf = np.mean([result['WinnerConfidence'] for result in results if result['PredictedWinner'] != 'Error'])
                    avg_method_conf = np.mean([result['MethodConfidence'] for result in results if result['Method'] != 'Error'])
                    
                    st.metric("Avg Winner Confidence", f"{round(avg_winner_conf, 1)}%")
                    st.metric("Avg Method Confidence", f"{round(avg_method_conf, 1)}%")
            
            # Option to download results
            csv = results_df[display_columns].to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='fight_predictions.csv',
                mime='text/csv',
            )
            
            # Enhanced visualization of batch results
            with st.expander("Batch Results Visualization", expanded=True):
                # Create simple charts showing overall stats
                col1, col2 = st.columns(2)
                
                with col1:
                    # Calculate how many red/blue corner wins were predicted
                    red_wins = sum(1 for result in results if result['WinnerColor'] == 'red' and result['PredictedWinner'] != 'Error')
                    blue_wins = sum(1 for result in results if result['WinnerColor'] == 'blue' and result['PredictedWinner'] != 'Error')
                    
                    # Create a pie chart of winner distribution
                    fig = go.Figure(go.Pie(
                        values=[red_wins, blue_wins],
                        labels=['Red Corner Wins', 'Blue Corner Wins'],
                        hole=0.4,
                        textinfo='label+percent',
                        marker=dict(colors=['#d20a0a', '#0a7ad2'])
                    ))
                    
                    fig.update_layout(
                        title="Predicted Winners",
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Calculate finish vs decision predictions
                    finishes = sum(1 for result in results if result['Method'] == 'Finish')
                    decisions = sum(1 for result in results if result['Method'] == 'Decision')
                    
                    # Create a pie chart of finish types
                    fig = go.Figure(go.Pie(
                        values=[finishes, decisions],
                        labels=['Finish', 'Decision'],
                        hole=0.4,
                        textinfo='label+percent',
                        marker=dict(colors=['#d20a0a', '#0a7ad2'])
                    ))
                    
                    fig.update_layout(
                        title="Predicted Finish Types",
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            # Add this missing except block to handle exceptions in batch predictions
            st.error(f"An error occurred during batch prediction: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()