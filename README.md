# UFC Fight Predictor 🥊

A machine learning application that predicts UFC fight outcomes based on fighter statistics and historical data.

![UFC Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/UFC_Logo.svg/2560px-UFC_Logo.svg.png)

## 📋 Features

- Predict fight winners with confidence percentages
- Predict fight finish methods (KO/TKO, Submission, Decision)
- Interactive UI for inputting fighter statistics
- Visual representation of prediction results
- Fighter comparison dashboard
- Comprehensive fighter stats input (physical attributes, fight history, performance metrics)

## 🚀 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Joblib

## 📊 How It Works

The UFC Fight Predictor uses trained machine learning models to analyze fighter statistics and predict:
1. The likely winner between two fighters
2. The probable method of victory (KO/TKO, Submission, or Decision)

The models take into account numerous factors such as:
- Fighter physical attributes (height, reach, age)
- Fighting style and statistics (striking accuracy, takedowns, etc.)
- Past performance (win/loss records, winning methods)
- Current form (win/loss streaks)
- Betting odds

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

1. Launch the application using the command above
2. Input fighter statistics for both the red and blue corner fighters
3. Add fight-specific information such as odds and weight class
4. Click "Predict Fight Outcome" to see the results
5. Review the visualizations and fighter comparison

## 📚 Project Structure

- `app.py`: Main Streamlit application
- `train_models.py`: Script to train prediction models
- `models/`: Directory containing trained models
- `utils/`: Utility functions for visualization and data processing
- `about.py`: About page content

## 📜 License

MIT License

## 👨‍💻 Author

[Turbash](https://github.com/Turbash)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!