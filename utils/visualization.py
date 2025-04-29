import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

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
    Plot prediction confidence instead of raw probabilities
    
    Parameters:
    -----------
    probabilities : array-like
        Probabilities from the model prediction
    classes : list
        Class labels
    prediction : int or str
        The predicted class
    title : str, optional
        Title for the plot
    """
    # Convert probabilities to confidence percentages
    confidence = [round(prob * 100, 1) for prob in probabilities]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, confidence, color=['#3498db', '#e74c3c', '#2ecc71'])
    
    # Highlight the predicted class
    pred_index = classes.index(prediction) if isinstance(prediction, str) else prediction
    bars[pred_index].set_color('#f39c12')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Outcome', fontsize=12)
    plt.ylabel('Confidence (%)', fontsize=12)
    
    # Add percentage labels on top of bars
    for i, confidence_val in enumerate(confidence):
        plt.text(i, confidence_val + 1, f"{confidence_val}%", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

def display_fight_prediction(winner_model, finish_model, finish_scaler, fight_data, red_fighter, blue_fighter):
    # Get features that the winner model was trained on
    if hasattr(winner_model, 'get_booster'):
        winner_features = winner_model.get_booster().feature_names
    else:
        # For models that don't have direct feature name access
        winner_features = [col for col in fight_data.columns if col not in [
            'RedDecOdds', 'BlueDecOdds', 'RSubOdds', 'BSubOdds', 'RKOOdds', 'BKOOdds'
        ]]
    
    # Filter fight_data to only include columns expected by the winner model
    winner_data = fight_data[winner_features]
    
    # Predict winner
    winner_pred = winner_model.predict(winner_data)[0]
    winner_proba = winner_model.predict_proba(winner_data)[0]
    
    # Determine winner name based on prediction (0 = Red, 1 = Blue)
    winner_name = red_fighter if winner_pred == 0 else blue_fighter
    
    # For finish model, we can use all features since the scaling will handle dimensionality
    finish_data = finish_scaler.transform(fight_data)
    
    # Predict finish type
    finish_pred = finish_model.predict(finish_data)[0]
    finish_proba = finish_model.predict_proba(finish_data)[0]
    
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