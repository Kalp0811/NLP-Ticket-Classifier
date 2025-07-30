# src/train_baseline.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from . import config

def train_baseline_model():
    """Trains and evaluates a TF-IDF + Logistic Regression baseline model."""
    print("--- Training Baseline Model ---")
    
    # 1. Load the sampled data
    df = pd.read_csv(config.SAMPLED_DATA_PATH)
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # 3. Create a model pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000)),
    ])
    
    # 4. Train the model
    print("Training Logistic Regression model...")
    pipeline.fit(X_train, y_train)
    
    # 5. Evaluate the model
    print("\nEvaluating baseline model...")
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=config.LABELS)
    print("Classification Report:")
    print(report)
    
    # 6. Save the model
    model_path = config.BASELINE_MODEL_DIR / "baseline_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\n✅ Baseline model saved to {model_path}")

if __name__ == "__main__":
    train_baseline_model()