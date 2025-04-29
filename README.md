# UFC Fight Prediction Application

This application predicts the outcome of UFC fights based on fighter statistics and fight details. It leverages machine learning models to predict both the winner of the fight (red or blue corner) and the type of finish (KO/TKO, Submission, or Decision).

## Features

- **Winner Prediction**: Predicts which fighter (red or blue corner) is likely to win.
- **Finish Type Prediction**: Predicts the method of victory (KO/TKO, Submission, or Decision).
- **Interactive Visualization**: Displays prediction confidence and other insights using Streamlit.
- **Customizable Models**: Includes scripts for training and fine-tuning machine learning models.

## Prerequisites

- Python 3.7+
- Required libraries (listed in `requirements.txt`)

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ufc-prediction-app.git
   cd ufc-prediction-app
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**:
   - Ensure the UFC dataset is in the `data` folder as `data/ufc-master.csv`.

4. **Train the Models**:
   ```bash
   python train_models.py
   ```

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## How It Works

1. **Data Preprocessing**:
   - Cleans and preprocesses the UFC dataset to handle missing values and outliers.
   - Scales features for better model performance.

2. **Machine Learning Models**:
   - Trains models (e.g., Logistic Regression, SVM) to predict fight outcomes.
   - Saves trained models for use in the application.

3. **Streamlit Application**:
   - Provides an interactive interface for users to input fight details and view predictions.
   - Displays prediction confidence and visualizations.

## Folder Structure

```
ufc-prediction-app/
│
├── app.py                 # Main Streamlit application
├── train_models.py        # Script for training models
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── data/                  # Folder for datasets
├── models/                # Folder for saved models
└── utils/                 # Utility scripts (preprocessing, visualization, etc.)
```

## Future Improvements

- Add support for additional fight statistics and features.
- Improve model accuracy with advanced algorithms (e.g., XGBoost, Neural Networks).
- Deploy the application online for public use.

---

**UFC Fight Prediction App - 2025**