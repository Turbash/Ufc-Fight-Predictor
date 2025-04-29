# UFC Fight Prediction Application

This application predicts the outcome of UFC fights based on fighter statistics and fight details. It uses machine learning models to predict both the winner of the fight (red or blue corner) and the type of finish (KO/TKO, Submission, or Decision).

## Prerequisites

- Python 3.7+
- Required libraries (listed in requirements.txt)

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure the UFC dataset is in the data folder as `data/ufc-master.csv`
4. Train the predictive models:
   ```
   python train_models.py
   ```
5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## How It Works

The application uses two machine learning models:

1. **Winner Prediction Model**: An XGBoost classifier that predicts whether the red or blue fighter will win.
2. **Finish Prediction Model**: A Logistic Regression classifier that predicts how the fight will end (KO/TKO, Submission, or Decision).

These models were trained on historical UFC fight data with features including fighter statistics, physical attributes, and fight context.

## Key Features

- Input fighter statistics for both red and blue corner fighters
- Visualize win and finish probabilities
- Easy-to-use interface for predicting fight outcomes
- Interactive visualization of prediction results

## Input Features

The application uses various fighter statistics as input, including:

- Fighter odds and expected values
- Win and loss streaks
- Striking statistics (significant strikes landed, accuracy)
- Grappling statistics (submission attempts, takedowns)
- Physical attributes (height, reach, weight)
- Age
- Previous win methods (KO, submission, decision)

## Dataset

The models are trained on the UFC Master dataset, which contains historical fight data with detailed statistics for both fighters and fight outcomes.

## Model Performance

The accuracy metrics for each model are displayed during training and saved with the models.

## Author

UFC Fight Prediction App - 2023