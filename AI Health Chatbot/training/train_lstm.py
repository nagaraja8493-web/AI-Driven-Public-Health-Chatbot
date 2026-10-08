"""
Training Pipeline for Deep Learning Component:
Embedding + Bi-LSTM (Bidirectional LSTM) with Dropout, Dense, and Softmax.
Uses identical stratified Train / Validation / Test splits (random_state=42)
for strict, unbiased ML vs DL comparison.
Generates:
- Training and Validation loss/accuracy curves
- Test set classification report, precision, recall, F1-score
- Confusion matrix
- Persisted .keras model and tokenizer artifacts.
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from chatbot.preprocessing import preprocess_pipeline

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "intents.csv")

def train_bilstm_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("STEP 1: Loading Dataset for Bi-LSTM Training")
    print("=" * 60)
    if not os.path.exists(DATA_PATH):
        from data.generate_dataset import generate_dataset
        generate_dataset(output_path=DATA_PATH)
        
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "intent"]).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(preprocess_pipeline)

    X = df["clean_text"].values
    y = df["intent"].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    classes = list(label_encoder.classes_)

    print("=" * 60)
    print("STEP 2: Stratified Split (80% Train, 10% Val, 10% Test)")
    print("=" * 60)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print("=" * 60)
    print("STEP 3: Tokenization & Sequence Padding")
    print("=" * 60)
    vocab_size = 3500
    max_len = 40
    embedding_dim = 128

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding="post", truncating="post")
    X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=max_len, padding="post", truncating="post")
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding="post", truncating="post")

    # One-hot encode targets for categorical crossentropy
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    print("=" * 60)
    print("STEP 4: Constructing Bi-LSTM Architecture")
    print("=" * 60)
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        SpatialDropout1D(0.25),
        Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    print("=" * 60)
    print("STEP 5: Training Bi-LSTM Model")
    print("=" * 60)
    best_model_path = os.path.join(MODELS_DIR, "dl_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    start_time = time.time()
    history = model.fit(
        X_train_seq, y_train_cat,
        validation_data=(X_val_seq, y_val_cat),
        epochs=25,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    training_time = round(time.time() - start_time, 4)
    print(f"Bi-LSTM training complete in {training_time:.4f} seconds.")

    print("=" * 60)
    print("STEP 6: Evaluating Bi-LSTM on Independent Test Set")
    print("=" * 60)
    # Load best saved weights
    best_model = tf.keras.models.load_model(best_model_path)
    y_test_probs = best_model.predict(X_test_seq)
    y_test_pred = np.argmax(y_test_probs, axis=1)

    y_val_probs = best_model.predict(X_val_seq)
    y_val_pred = np.argmax(y_val_probs, axis=1)

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

    print("\nDetailed Classification Report (Bi-LSTM):")
    print(classification_report(y_test, y_test_pred, target_names=classes, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix - Bi-LSTM (Test Set)")
    plt.xlabel("Predicted Intent")
    plt.ylabel("True Intent")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "dl_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_path}")

    # Training Curves (Accuracy & Loss)
    epochs_range = range(1, len(history.history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, history.history["accuracy"], "b-o", label="Training Accuracy")
    ax1.plot(epochs_range, history.history["val_accuracy"], "g-s", label="Validation Accuracy")
    ax1.set_title("Bi-LSTM Model Accuracy across Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epochs_range, history.history["loss"], "r-o", label="Training Loss")
    ax2.plot(epochs_range, history.history["val_loss"], "m-s", label="Validation Loss")
    ax2.set_title("Bi-LSTM Model Loss across Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Categorical Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    curves_path = os.path.join(MODELS_DIR, "dl_training_curves.png")
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Training curves saved to {curves_path}")

    # Save metrics
    metrics = {
        "model": "Bi-LSTM (Embedding + Bi-LSTM + Softmax)",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "training_time_seconds": training_time,
        "epochs_trained": len(history.history["loss"]),
        "validation_accuracy": round(float(val_accuracy), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "test_precision": round(float(test_precision), 4),
        "test_recall": round(float(test_recall), 4),
        "test_f1_score": round(float(test_f1), 4),
        "classes": classes
    }
    metrics_path = os.path.join(MODELS_DIR, "dl_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save artifacts
    joblib.dump(tokenizer, os.path.join(MODELS_DIR, "tokenizer.pkl"))
    joblib.dump({"max_len": max_len, "vocab_size": vocab_size}, os.path.join(MODELS_DIR, "dl_params.pkl"))
    print("All DL artifacts saved to models/ directory.")

    return metrics

if __name__ == "__main__":
    train_bilstm_model()
