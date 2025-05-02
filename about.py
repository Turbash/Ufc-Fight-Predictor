import streamlit as st

def show_about_page():
    st.markdown("<h1 style='text-align: center; color: #d20a0a;'>About UFC Fight Predictor</h1>", unsafe_allow_html=True)
    
    # Introduction section
    st.markdown("""
    <div style="background: linear-gradient(145deg, #1a1e25 0%, #2c3e50 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #ffc107;">
        <h2 style="color: #ffc107;">Introduction</h2>
        <p style="color: #ffffff; font-size: 1.1rem; line-height: 1.6;">
            UFC Fight Predictor is a machine learning application designed to predict the outcomes of UFC fights. 
            Using historical fight data and advanced algorithms, the system can predict both the winner of a fight and 
            whether the fight will end via finish (KO/TKO/Submission) or go to a decision.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works section
    st.markdown("""
    <div style="background: linear-gradient(145deg, #1a1e25 0%, #2c3e50 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #d20a0a;">
        <h2 style="color: #d20a0a;">How It Works</h2>
        <p style="color: #ffffff; font-size: 1.1rem; line-height: 1.6;">
            The prediction system uses ensemble machine learning models trained on thousands of historical UFC fights.
            These models analyze various factors including:
        </p>
        <ul style="color: #ffffff; font-size: 1.1rem; line-height: 1.6;">
            <li>Fighter physical attributes (height, reach, age)</li>
            <li>Fight records and win streaks</li>
            <li>Performance statistics (striking accuracy, takedowns, etc.)</li>
            <li>Betting odds and market sentiment</li>
            <li>Fighting styles and matchup dynamics</li>
        </ul>
        <p style="color: #ffffff; font-size: 1.1rem; line-height: 1.6;">
            The system then processes this information to generate probability-based predictions for fight outcomes.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("""
    <div style="background: linear-gradient(145deg, #1a1e25 0%, #2c3e50 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #0a7ad2;">
        <h2 style="color: #0a7ad2;">Key Features</h2>
        <ul style="color: #ffffff; font-size: 1.1rem; line-height: 1.6;">
            <li><strong>Single Fight Prediction:</strong> Input fighter details to get winner and finish predictions</li>
            <li><strong>Batch Predictions:</strong> Upload CSV files with multiple fights for efficient analysis</li>
            <li><strong>Probability-Based Results:</strong> See confidence levels for each prediction</li>
            <li><strong>Visual Analytics:</strong> Interactive charts and visualizations of prediction factors</li>
            <li><strong>Fighter Comparison:</strong> Side-by-side comparison of fighter attributes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Limitations & Disclaimer
    st.markdown("""
    <div style="background: linear-gradient(145deg, #1a1e25 0%, #2c3e50 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #ffc107;">
        <h2 style="color: #ffc107;">Limitations & Disclaimer</h2>
        <p style="color: #ffffff; font-size: 1.1rem; line-height: 1.6;">
            While this application uses advanced machine learning techniques, please keep in mind:
        </p>
        <ul style="color: #ffffff; font-size: 1.1rem; line-height: 1.6;">
            <li>No prediction system can account for all variables in combat sports</li>
            <li>Last-minute changes (injuries, weight cut issues) may not be reflected</li>
            <li>The emotional and psychological aspects of fighting are difficult to quantify</li>
            <li>This tool is for entertainment and educational purposes only</li>
            <li>We do not recommend making betting decisions solely based on these predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Technology stack
    st.markdown("""
    <div style="background: linear-gradient(145deg, #1a1e25 0%, #2c3e50 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #28a745;">
        <h2 style="color: #28a745;">Technology Stack</h2>
        <p style="color: #ffffff; font-size: 1.1rem; line-height: 1.6;">
            This application is built using:
        </p>
        <ul style="color: #ffffff; font-size: 1.1rem; line-height: 1.6;">
            <li><strong>Python:</strong> Core programming language</li>
            <li><strong>Streamlit:</strong> Web application framework</li>
            <li><strong>Scikit-learn:</strong> Machine learning algorithms</li>
            <li><strong>XGBoost:</strong> Gradient boosting implementation</li>
            <li><strong>Pandas & NumPy:</strong> Data manipulation</li>
            <li><strong>Plotly & Matplotlib:</strong> Data visualization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact & Feedback
    st.markdown("""
    <div style="background: linear-gradient(145deg, #1a1e25 0%, #2c3e50 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #d20a0a;">
        <h2 style="color: #d20a0a;">Contact & Feedback</h2>
        <p style="color: #ffffff; font-size: 1.1rem; line-height: 1.6;">
            We're constantly working to improve our predictions. If you have suggestions, comments, 
            or would like to contribute to the project, please visit:
        </p>
        <p style="text-align: center; margin-top: 20px;">
            <a href="https://github.com/Turbash/Ufc-Fight-Predictor" target="_blank" style="background-color: #ffc107; color: #000; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">GitHub Repository</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; color: #7f8c8d; font-size: 0.9rem;">
        <p>© 2025 UFC Fight Predictor | Not affiliated with UFC® or Zuffa LLC</p>
    </div>
    """, unsafe_allow_html=True)

# Test the about page when run directly
if __name__ == "__main__":
    show_about_page()
