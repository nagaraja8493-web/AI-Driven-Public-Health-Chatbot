"""
Prediction Engine for AI Public Health Chatbot.
Loads TF-IDF + Logistic Regression (ML) and Tokenizer + Bi-LSTM (DL) models.
Performs emergency checking, inference, confidence evaluation, and response mapping.
"""

import os
import joblib
import numpy as np
from chatbot.preprocessing import preprocess_pipeline
from chatbot.response import check_emergency, get_response_for_intent, EMERGENCY_RESPONSE

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

class HealthChatbotPredictor:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.ml_model = None
        self.tfidf_vectorizer = None
        self.dl_model = None
        self.dl_tokenizer = None
        self.label_encoder = None
        self.max_seq_len = 40
        self.models_loaded = False
        self.load_models()

    def load_models(self):
        """Attempts to load pre-trained ML and DL models if available."""
        # Load ML components
        ml_path = os.path.join(self.models_dir, "ml_model.pkl")
        tfidf_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
        le_path = os.path.join(self.models_dir, "label_encoder.pkl")

        if os.path.exists(ml_path) and os.path.exists(tfidf_path):
            try:
                self.ml_model = joblib.load(ml_path)
                self.tfidf_vectorizer = joblib.load(tfidf_path)
                print("Loaded ML model (Logistic Regression) and TF-IDF vectorizer.")
            except Exception as e:
                print(f"Error loading ML model: {e}")

        if os.path.exists(le_path):
            try:
                self.label_encoder = joblib.load(le_path)
            except Exception as e:
                print(f"Error loading label encoder: {e}")

        # Load DL components
        dl_path = os.path.join(self.models_dir, "dl_model.keras")
        tok_path = os.path.join(self.models_dir, "tokenizer.pkl")
        params_path = os.path.join(self.models_dir, "dl_params.pkl")

        if os.path.exists(dl_path) and os.path.exists(tok_path):
            try:
                import tensorflow as tf
                self.dl_model = tf.keras.models.load_model(dl_path)
                self.dl_tokenizer = joblib.load(tok_path)
                if os.path.exists(params_path):
                    params = joblib.load(params_path)
                    self.max_seq_len = params.get("max_len", 40)
                print("Loaded DL model (Bi-LSTM) and Tokenizer.")
            except Exception as e:
                print(f"Error loading DL model: {e}")

        self.models_loaded = (self.ml_model is not None or self.dl_model is not None)

    def predict_ml(self, text: str):
        """Inference using TF-IDF + Logistic Regression."""
        if not self.ml_model or not self.tfidf_vectorizer:
            raise RuntimeError("ML model or TF-IDF vectorizer is not loaded.")
        
        cleaned = preprocess_pipeline(text)
        features = self.tfidf_vectorizer.transform([cleaned])
        probs = self.ml_model.predict_proba(features)[0]
        max_idx = int(np.argmax(probs))
        top_indices = np.argsort(probs)[::-1][:3]
        if self.label_encoder:
            intent = str(self.label_encoder.inverse_transform([max_idx])[0])
            top_intents = [
                {"intent": str(self.label_encoder.inverse_transform([int(i)])[0]), "confidence": round(float(probs[i]), 4)}
                for i in top_indices
            ]
        else:
            intent = str(self.ml_model.classes_[max_idx])
            top_intents = [
                {"intent": str(self.ml_model.classes_[int(i)]), "confidence": round(float(probs[i]), 4)}
                for i in top_indices
            ]
        confidence = float(probs[max_idx])

        return intent, confidence, top_intents

    def predict_dl(self, text: str):
        """Inference using Tokenizer + Bi-LSTM."""
        if not self.dl_model or not self.dl_tokenizer or not self.label_encoder:
            # Fall back to ML if DL model is not available
            if self.ml_model:
                return self.predict_ml(text)
            raise RuntimeError("DL model is not loaded.")

        from tensorflow.keras.preprocessing.sequence import pad_sequences
        cleaned = preprocess_pipeline(text)
        seq = self.dl_tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=self.max_seq_len, padding="post", truncating="post")
        probs = self.dl_model.predict(padded, verbose=0)[0]
        max_idx = int(np.argmax(probs))
        intent = str(self.label_encoder.inverse_transform([max_idx])[0])
        confidence = float(probs[max_idx])

        top_indices = np.argsort(probs)[::-1][:3]
        top_intents = [
            {"intent": str(self.label_encoder.inverse_transform([int(i)])[0]), "confidence": round(float(probs[i]), 4)}
            for i in top_indices
        ]

        return intent, confidence, top_intents

    def get_chat_response(self, text: str, model_type: str = "ml", confidence_threshold: float = 0.60) -> dict:
        """
        Full prediction and response pipeline:
        1. Emergency safety layer check
        2. Model inference (ML or DL)
        3. Confidence check & fallback
        4. Knowledge Base response retrieval
        """
        # Step 1: Emergency safety layer
        if check_emergency(text):
            return {
                "intent": "emergency",
                "confidence": 1.0,
                "response": EMERGENCY_RESPONSE,
                "model_used": "safety_rules",
                "is_fallback": False,
                "is_emergency": True,
                "top_intents": [{"intent": "emergency", "confidence": 1.0}]
            }

        # Step 2: Intent classification
        try:
            if model_type.lower() == "dl" and self.dl_model is not None:
                predicted_intent, confidence, top_intents = self.predict_dl(text)
                active_model = "Bi-LSTM (DL)"
            else:
                predicted_intent, confidence, top_intents = self.predict_ml(text)
                active_model = "Logistic Regression (ML)"
        except Exception as e:
            # Return graceful fallback if models are not yet trained
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "response": f"The predictive model is currently initializing. ({str(e)})",
                "model_used": model_type,
                "is_fallback": True,
                "is_emergency": False,
                "top_intents": []
            }

        # Step 3: Response retrieval & confidence evaluation
        response_text, is_fallback = get_response_for_intent(
            predicted_intent, confidence, threshold=confidence_threshold
        )

        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "response": response_text,
            "model_used": active_model,
            "is_fallback": is_fallback,
            "is_emergency": (predicted_intent == "emergency"),
            "top_intents": top_intents
        }

# Global singleton predictor
_predictor_instance = None

def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = HealthChatbotPredictor()
    return _predictor_instance
