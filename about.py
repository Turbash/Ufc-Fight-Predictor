import streamlit as st

def show_about_page():
    st.title("About UFC Prediction Model")
    
    st.markdown("""
    ## Project Overview
    
    The UFC Prediction App is a machine learning-based tool that predicts the outcomes of UFC fights. 
    The app analyzes historical fight data and fighter statistics to make predictions on:
    
    1. **Who will win the fight** - Red corner or Blue corner fighter
    2. **How the fight will end** - KO/TKO, Submission, or Decision
    
    ## Data and Models
    
    ### Data Source
    
    The model is trained on a comprehensive dataset of UFC fights including:
    - Fighter statistics (height, reach, weight, age)
    - Fighting style and stance
    - Win/loss records
    - Performance metrics (strikes landed, takedowns, etc.)
    - Betting odds
    
    ### Machine Learning Models
    
    Two separate models are used:
    
    #### Winner Prediction Model
    - **Algorithm**: XGBoost Classifier
    - **Features**: Fighter statistics, betting odds, and performance metrics
    - **Output**: Probability of each fighter winning
    
    #### Finish Type Prediction Model
    - **Algorithm**: Logistic Regression
    - **Features**: Similar features as the winner model plus specialized odds for KO/SUB
    - **Output**: Probability of fight ending in KO/TKO, Submission, or Decision
    
    ## How to Use
    
    1. Enter the statistics for both fighters
    2. Review the prediction results showing the likely winner and finish type
    3. The model will display confidence percentages for each outcome
    
    ## Limitations
    
    This model is based on historical data and doesn't account for:
    - Last-minute changes in fighter conditions
    - Psychological factors and game plans
    - New techniques a fighter may have developed
    
    ## Development
    
    The app was developed using:
    - Python for data processing and modeling
    - Scikit-learn and XGBoost for machine learning
    - Streamlit for the web interface
    - Pandas for data manipulation
    - Matplotlib and Seaborn for visualizations
    """)
    
    st.subheader("Contact")
    st.markdown("For questions or suggestions about this project, please contact us.")
    
    # GitHub repository link
    st.markdown("[GitHub Repository](https://github.com/yourusername/ufc-prediction-app)")
    
    # Version info
    st.sidebar.info("Version: 1.0.0")
