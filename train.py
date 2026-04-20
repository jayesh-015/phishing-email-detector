import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from utils import clean_text, extract_custom_features, get_bert_embeddings

def main():
    print("\n" + "="*60)
    print("  PHISHING EMAIL DETECTOR - MODEL TRAINING")
    print("="*60)
    
    print("\n[1/5] Loading dataset...")
    try:
        df = pd.read_csv('dataset/email_data.csv')
        print(f"      ✓ Loaded {len(df)} emails")
    except FileNotFoundError:
        print("      ✗ Error: dataset/email_data.csv not found")
        return
    
    print("\n[2/5] Cleaning and preprocessing text...")
    df['clean_text'] = df['text'].apply(clean_text)
    print(f"      ✓ Text cleaned")

    print("\n[3/5] Extracting cybersecurity heuristic features...")
    print("       - URL count")
    print("       - Email length")
    print("       - Risk score (keywords)")
    print("       - HTML content detection")
    custom_features = extract_custom_features(df).values
    print(f"      ✓ {custom_features.shape[1]} features extracted")

    print("\n[4/5] Generating BERT embeddings...")
    bert_embeddings = get_bert_embeddings(df['clean_text'])
    print(f"      ✓ {bert_embeddings.shape[1]}-dimensional embeddings created")

    # Combine features
    X_combined = np.hstack((custom_features, bert_embeddings))
    y = df['label'].to_numpy()
    print(f"\n      Total features: {X_combined.shape[1]}")
    print(f"      Phishing emails: {sum(y)} | Safe emails: {len(y) - sum(y)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)
    print(f"      Training set: {len(X_train)} | Test set: {len(X_test)}")

    print("\n[5/5] Training Random Forest classifier...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    classifier.fit(X_train, y_train)
    print("      ✓ Training complete")

    # Evaluate
    print("\n" + "="*60)
    print("  EVALUATION METRICS")
    print("="*60)
    
    y_pred = classifier.predict(X_test)
    
    print(f"\nAccuracy:   {accuracy_score(y_test, y_pred):.2%}")
    print(f"Precision:  {precision_score(y_test, y_pred):.2%}")
    print(f"Recall:     {recall_score(y_test, y_pred):.2%}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0,0]} | False Positives: {cm[0,1]}")
    print(f"  False Negatives: {cm[1,0]} | True Positives:  {cm[1,1]}")

    # Save model
    print("\n" + "="*60)
    import os
    os.makedirs('model', exist_ok=True)
    joblib.dump(classifier, 'model/phishing_rf_ensemble.pkl')
    print("✓ Model saved to model/phishing_rf_ensemble.pkl")
    print("="*60)
    print("\nTraining complete! You can now run: python app.py")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()