"""
Training Pipeline for Machine Learning Component:
TF-IDF Vectorization + Logistic Regression.
Adheres strictly to ml-best-practices:
- Train / Validation / Test stratified split
- Strict featurization ordering (TF-IDF fitted ONLY on training split)
- Evaluation on test split: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Real, un-fabricated metrics persisted to JSON & serialized model artifacts.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from chatbot.preprocessing import preprocess_pipeline

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "intents.csv")

def train_ml_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("STEP 1: Loading and Preprocessing Dataset")
    print("=" * 60)
    if not os.path.exists(DATA_PATH):
        from data.generate_dataset import generate_dataset
        print("Dataset not found. Generating dataset...")
        generate_dataset(output_path=DATA_PATH)
        
    df = pd.read_csv(DATA_PATH)
    print(f"Total samples loaded: {len(df)}")
    print(f"Unique intents: {df['intent'].nunique()}")

    # Clean missing values
    df = df.dropna(subset=["text", "intent"]).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(preprocess_pipeline)

    X = df["clean_text"].values
    y = df["intent"].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = list(label_encoder.classes_)

    print("=" * 60)
    print("STEP 2: Stratified Train / Validation / Test Split (80% / 10% / 10%)")
    print("=" * 60)
    # First split: 80% train, 20% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    # Second split: 10% val, 10% test from temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Training samples:   {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test samples:       {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    print("=" * 60)
    print("STEP 3: Strict Featurization Ordering (Fitting TF-IDF on Train Only)")
    print("=" * 60)
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=4000,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"Vocabulary size: {len(tfidf.vocabulary_)} features")

    print("=" * 60)
    print("STEP 4: Training Logistic Regression Model")
    print("=" * 60)
    start_time = time.time()
    lr_model = LogisticRegression(
        C=2.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    )
    lr_model.fit(X_train_tfidf, y_train)
    training_time = round(time.time() - start_time, 4)
    print(f"Model trained successfully in {training_time:.4f} seconds.")

    print("=" * 60)
    print("STEP 5: Model Evaluation on Independent Test Set")
    print("=" * 60)
    y_test_pred = lr_model.predict(X_test_tfidf)
    y_val_pred = lr_model.predict(X_val_tfidf)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy:       {test_accuracy:.4f}")
    print(f"Test Precision:      {test_precision:.4f}")
    print(f"Test Recall:         {test_recall:.4f}")
    print(f"Test F1 Score:       {test_f1:.4f}")

    print("\nDetailed Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=classes, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix - Logistic Regression (Test Set)")
    plt.xlabel("Predicted Intent")
    plt.ylabel("True Intent")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "ml_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_path}")

    # Save metrics
    metrics = {
        "model": "Logistic Regression (TF-IDF)",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "training_time_seconds": training_time,
        "validation_accuracy": round(float(val_accuracy), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "test_precision": round(float(test_precision), 4),
        "test_recall": round(float(test_recall), 4),
        "test_f1_score": round(float(test_f1), 4),
        "classes": classes
    }
    metrics_path = os.path.join(MODELS_DIR, "ml_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save artifacts
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "ml_model.pkl"))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print("All ML artifacts saved to models/ directory.")

    return metrics

if __name__ == "__main__":
    train_ml_model()
