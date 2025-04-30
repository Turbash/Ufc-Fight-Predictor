import streamlit as st

def show_about_page():
    st.markdown("<h1 style='text-align: center; color: #E50914;'>About UFC Fight Predictor</h1>", unsafe_allow_html=True)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/UFC_Logo.svg/2560px-UFC_Logo.svg.png", width=300)
    
    st.markdown("""
    ## Project Overview
    
    UFC Fight Predictor is a machine learning application designed to predict the outcomes of UFC fights. Using historical fighter data, statistical analysis, and machine learning algorithms, this tool provides insights into potential winners and how fights might end.
    
    ## How It Works
    
    The application uses two primary machine learning models:
    
    1. **Winner Prediction Model**: Simple Logistic Regression model that analyzes fighter statistics to determine the likely winner between two opponents. Achieves ~66% accuracy.
    
    2. **Finish Type Model**: Another Logistic Regression model that predicts how the fight will end - KO/TKO, Submission, or Decision. Achieves ~55% accuracy.
    
    The models are trained on historical UFC fight data, considering factors such as:
    
    - Physical attributes (height, reach, weight)
    - Technical performance (strike accuracy, takedowns, submissions)
    - Fight history (wins, losses, methods of victory)
    - Current form (win/loss streaks)
    - Betting odds and market expectations
    
    ## Data Sources
    
    This project utilizes data compiled from various public UFC statistics sources. The dataset includes detailed information about fighters, their performances, and fight outcomes going back several years.
    
    ## Model Performance
    
    The current version of our models achieves:
    
    - Winner prediction accuracy: ~66%
    - AUC-ROC for winner prediction: 0.71
    - Fight finish method accuracy: ~55%
    - AUC-ROC for finish method (multiclass): 0.65
    
    These accuracy rates are comparable to or better than the accuracy of betting odds in predicting outcomes.
    
    ## Limitations
    
    While our models provide valuable insights, they have limitations:
    
    - They cannot account for undisclosed injuries or personal factors
    - Last-minute fight changes may affect prediction accuracy
    - The unpredictable nature of combat sports means upsets will happen
    
    ## Future Improvements
    
    We're constantly working to improve the models with:
    
    - More granular fight data
    - Advanced feature engineering
    - Incorporation of momentum and trend analysis
    - Consideration of stylistic matchups
    
    ## Credits
    
    Developed by [Turbash](https://github.com/Turbash)
    
    ## Feedback & Contributions
    
    Feedback, bug reports, and contributions are welcome on the [GitHub repository](https://github.com/Turbash/Ufc-Fight-Predictor).
    """)
    
    # Add disclaimer
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This application is intended for entertainment and informational purposes only. 
    It should not be used as the sole basis for betting or gambling decisions. Always gamble responsibly.
    """)
    
    # Add contact section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Contact")
    st.sidebar.markdown("[GitHub](https://github.com/Turbash)")
