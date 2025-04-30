import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import numpy as np

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

def display_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print("Classification Report:\n", report)
    
def plot_prediction_confidence(probabilities, classes, prediction, title="Prediction Confidence"):
    """
    Plot the confidence scores for a prediction.
    
    Args:
        probabilities: Array of prediction probabilities
        classes: Class labels
        prediction: The predicted class
        title: Plot title
    """
    # Ensure probabilities are proper values between 0 and 1
    # Sometimes models return extreme values that need normalization
    probabilities = np.array(probabilities)
    
    # If all probabilities are very close to 0 or 1, adjust them
    if np.max(probabilities) > 0.99 and np.min(probabilities) < 0.01:
        # Apply softmax to renormalize extreme probabilities
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        
        # Convert probabilities to logits (approximate) and apply softmax
        logits = np.log(probabilities / (1 - probabilities + 1e-10))
        probabilities = softmax(logits)
    
    # Convert probabilities to percentages
    confidence = [round(prob * 100, 1) for prob in probabilities]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, confidence, color=['lightblue' if c != prediction else 'blue' for c in classes])
    
    # Highlight the predicted class
    for i, bar in enumerate(bars):
        if classes[i] == prediction:
            bar.set_color('blue')
    
    # Add labels and title
    plt.xlabel('Outcome')
    plt.ylabel('Confidence (%)')
    plt.title(title)
        
    # Add percentage labels on top of bars
    for i, confidence_val in enumerate(confidence):
        plt.text(i, confidence_val + 1, f"{confidence_val}%", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

def display_fight_prediction(winner_model, finish_model, finish_scaler, fight_data, red_fighter, blue_fighter):
    # Get features that the winner model was trained on
    try:
        import joblib
        winner_features = joblib.load('models/winner_features.pkl')
    except Exception as e:
        # Fallback if feature list can't be loaded
        if hasattr(winner_model, 'get_booster'):
            winner_features = winner_model.get_booster().feature_names
        else:
            # For models that don't have direct feature name access
            winner_features = [col for col in fight_data.columns if col not in [
                'RedDecOdds', 'BlueDecOdds', 'RSubOdds', 'BSubOdds', 'RKOOdds', 'BKOOdds'
            ]]
    
    # Ensure all required columns exist in fight_data
    for col in winner_features:
        if col not in fight_data.columns:
            fight_data[col] = 0  # Default to 0 for missing columns
    
    # Filter fight_data to only include columns expected by the winner model
    winner_data = fight_data[winner_features]
    
    # Apply scaling to winner data using the saved scaler
    try:
        # Load the winner scaler
        winner_scaler = joblib.load('models/winner_scaler.pkl')
        winner_data_scaled = winner_scaler.transform(winner_data)
        
        # Predict using scaled data
        winner_pred = winner_model.predict(winner_data_scaled)[0]
        winner_proba = winner_model.predict_proba(winner_data_scaled)[0]
        
        # Apply temperature scaling to adjust overconfident predictions
        if np.max(winner_proba) > 0.95:
            temperature = 2.0  # Higher temperature gives softer probabilities
            logits = np.log(winner_proba / (1 - winner_proba + 1e-10))
            softmax_probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
            winner_proba = softmax_probs
    except:
        # Fall back to unscaled prediction if scaler can't be loaded
        winner_pred = winner_model.predict(winner_data)[0]
        winner_proba = winner_model.predict_proba(winner_data)[0]
    
    # Determine winner name based on prediction (0 = Red, 1 = Blue)
    winner_name = red_fighter if winner_pred == 0 else blue_fighter
    
    # For finish model, use features and apply scaling
    try:
        finish_features = joblib.load('models/finish_features.pkl')
        finish_data = fight_data[finish_features]
        
        # Load and apply finish scaler
        finish_scaler = joblib.load('models/finish_scaler.pkl')
        finish_data_scaled = finish_scaler.transform(finish_data)
        
        # Predict using scaled data
        finish_pred = finish_model.predict(finish_data_scaled)[0]
        finish_proba = finish_model.predict_proba(finish_data_scaled)[0]
    except:
        # Fall back to unscaled prediction if there's an issue
        if finish_features is None:
            finish_data = winner_data  # Use same features as winner if finish features not found
        else:
            finish_data = fight_data[finish_features]
            
        finish_pred = finish_model.predict(finish_data)[0]
        finish_proba = finish_model.predict_proba(finish_data)[0]
    
    # Apply similar temperature scaling to finish probabilities if needed
    if np.max(finish_proba) > 0.95:
        temperature = 2.0
        logits = np.log(finish_proba / (1 - finish_proba + 1e-10))
        softmax_probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        finish_proba = softmax_probs

    finish_types = ['KO/TKO', 'Submission', 'Decision']
    finish_type = finish_types[finish_pred]
    
    # Display predictions with proper styling
    print("\n" + "="*50)
    print(f"🥊 FIGHT PREDICTION: {red_fighter} vs {blue_fighter} 🥊")
    print("="*50)
    
    print(f"\n🏆 PREDICTED WINNER: {winner_name}")
    print(f"📊 CONFIDENCE: {round(winner_proba[winner_pred] * 100, 1)}%")
    
    print(f"\n🔄 PREDICTED FINISH: {finish_type}")
    print(f"📊 CONFIDENCE: {round(finish_proba[finish_pred] * 100, 1)}%")
    print("\n" + "="*50)
    
    # Plot visual representations
    winner_classes = [red_fighter, blue_fighter]
    plot_prediction_confidence(winner_proba, winner_classes, winner_classes[winner_pred], 
                              "Winner Prediction Confidence")
    
    plot_prediction_confidence(finish_proba, finish_types, finish_types[finish_pred],
                              "Finish Type Prediction Confidence")