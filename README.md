# UFC Fight Predictor 🥊

A machine learning application that predicts UFC fight outcomes based on fighter statistics and historical data.

![UFC Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/UFC_Logo.svg/2560px-UFC_Logo.svg.png)

## 📋 Features

- Predict fight winners with confidence percentages
- Predict fight outcomes (Finish or Decision)
- Interactive modern UI with beautiful visualizations
- Batch prediction capabilities via CSV upload
- Fighter comparison dashboard with modern charts
- Comprehensive fighter stats input (physical attributes, fight history, performance metrics)
- Betting odds integration

## 🚀 Technologies Used

- Python 3.9+
- Streamlit
- Pandas & NumPy
- Plotly & Matplotlib
- Scikit-learn
- XGBoost
- Joblib

## 📊 How It Works

The UFC Fight Predictor uses ensemble machine learning models to analyze fighter statistics and predict:
1. The likely winner between two fighters (~67% accuracy)
2. Whether the fight will end via finish or decision (~58% accuracy)

The models take into account numerous factors such as:
- Fighter physical attributes (height, reach, age)
- Fighting style and statistics (striking accuracy, takedowns, etc.)
- Past performance (win/loss records, winning methods)
- Current form (win/loss streaks)
- Betting odds and market sentiment

## 🛠️ Installation

1. Clone the repository:
   ```
   git clone https://github.com/Turbash/Ufc-Fight-Predictor.git
   cd Ufc-Fight-Predictor
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Train the models (if not already included):
   ```
   python train_models.py
   ```

5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## 📝 Usage

### Single Fight Prediction
1. Input fighter statistics for both the red and blue corner fighters
2. Add fight-specific information such as odds and rounds
3. Click "Predict Fight Outcome" to see the results
4. Review the visualizations and fighter comparison charts

### Batch Predictions
1. Navigate to the "Batch Predictions" page
2. Upload a CSV file containing multiple fights (template available)
3. Review the prediction results table
4. Analyze the batch results visualizations
5. Download the predictions as CSV

## 📈 Model Performance

Our models currently achieve:
- **Winner prediction**: ~67% accuracy (AUC-ROC: 0.72)
- **Finish prediction**: ~58% accuracy (AUC-ROC: 0.65)

## 📚 Project Structure

- `app.py`: Main Streamlit application
- `train_models.py`: Script to train prediction models
- `models/`: Directory containing trained models
- `utils/`: Utility functions for visualization, data processing, and feature importance
- `about.py`: About page content with project information

## 📜 License

MIT License

## 👨‍💻 Author

[Turbash](https://github.com/Turbash)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## ⚠️ Disclaimer

This application is for entertainment and educational purposes only. Predictions should not be used as the sole basis for betting decisions. UFC® is a registered trademark of Zuffa LLC. This project is not affiliated with UFC® or Zuffa LLC.