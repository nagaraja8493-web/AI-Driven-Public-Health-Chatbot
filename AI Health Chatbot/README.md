# AI-Driven Public Health Chatbot
### Leveraging Machine Learning (TF-IDF + Logistic Regression) and Deep Learning (Bi-LSTM)

An intelligent conversational public-health awareness system designed to provide evidence-based health guidance, symptom awareness, and preventive advice. Built with an ethical clinical safety layer that prevents direct diagnostic claims, enforces confidence score thresholds, and immediately redirects emergency queries to India's **112** unified national helpline.

---

## 🌟 Key Features

1. **Dual AI Intent Classification Engine:**
   - **Machine Learning (Baseline):** TF-IDF n-gram vectorization with regularized multinomial Logistic Regression.
   - **Deep Learning:** Learned Word Embeddings + Bidirectional LSTM (Bi-LSTM) recurrent neural network with SpatialDropout and Softmax.
   - **Model Switcher:** Toggle between ML and Deep Learning in real time inside the chat UI.

2. **Ethical Clinical Safety & Knowledge Base:**
   - **No Direct Diagnosis:** The system provides general disease awareness and preventive lifestyle guidelines; it never claims to diagnose conditions or prescribe medications.
   - **Confidence Threshold Check (Default 60%):** If model confidence falls below the threshold, the system provides an ethical fallback and advises consulting a physician.
   - **Immediate Emergency Safety Layer:** Scans for critical symptoms (crushing chest pain, severe breathlessness, stroke signs, heavy bleeding) and immediately bypasses AI inference to provide India's **112** and **102/108** emergency numbers.

3. **Multilingual & Transliteration Support:**
   - Preprocessing pipeline handles English and transliterated Kannada terms (e.g. *jvara, seetha, kemmu, sakkare kayile, rakta ottada*).

4. **Voice & Audio Integration:**
   - 🎤 **Voice Input:** Dictate queries using browser Web Speech Recognition.
   - 🔊 **Text-to-Speech:** Click to read bot responses aloud via Web SpeechSynthesis.

5. **Searchable Health Topics Directory:**
   - 16+ verified health topics with filters, prevention guides, and "When to Seek Professional Medical Care" modals.

6. **Admin Analytics Dashboard:**
   - Real-time statistics: Total users, total questions, average confidence, emergency trigger count.
   - Interactive Chart.js charts: Most asked health topics, intent distribution, ML vs DL comparison, and query volume timeline.
   - Audit table for low-confidence queries (< 60%).

---

## 🏗️ System Architecture

```
                       ┌─────────────────────────┐
                       │   User Health Question  │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ NLP Text Preprocessing  │
                       │ (Clean, Expand, Kannada)│
                       └────────────┬────────────┘
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
            ┌──────────────────────┐  ┌──────────────────────┐
            │ Machine Learning (ML)│  │ Deep Learning (DL)   │
            │ TF-IDF Vectorizer    │  │ Tokenizer + Embedding│
            │ Logistic Regression  │  │ Bi-LSTM + Softmax    │
            └──────────┬───────────┘  └──────────┬───────────┘
                       └────────────┬────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │ Intent & Confidence (C) │
                       └────────────┬────────────┘
                                    │
                   ┌────────────────┴────────────────┐
            [C < 0.60]                               [C >= 0.60]
                   ▼                                 ▼
        ┌───────────────────────┐         ┌───────────────────────┐
        │ Ethical Fallback Safe │         │ Health Knowledge Base │
        │ Advisory + Rephrase   │         │ Verified Guidance     │
        └──────────┬────────────┘         └──────────┬────────────┘
                   └────────────────┬────────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │ Formatted Chat Response │
                       │ + Clinical Disclaimer   │
                       └─────────────────────────┘
```

---

## 📁 Project Directory Structure

```
AI-Health-Chatbot/
│
├── app.py                     # Primary Flask web server & REST APIs
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation & instructions
│
├── data/
│   ├── generate_dataset.py    # Dataset generator (5,843 questions across 18 intents)
│   ├── intents.csv            # Intent classification dataset
│   └── health_topics.csv      # Verified health topics and prevention guide
│
├── models/
│   ├── ml_model.pkl           # Trained Logistic Regression classifier
│   ├── tfidf_vectorizer.pkl   # Fitted TF-IDF vectorizer
│   ├── dl_model.keras         # Trained Bi-LSTM Keras neural network
│   ├── tokenizer.pkl          # Keras Tokenizer
│   ├── label_encoder.pkl      # Sklearn LabelEncoder
│   ├── ml_metrics.json        # Real ML test metrics & evaluation
│   ├── dl_metrics.json        # Real DL test metrics & evaluation
│   ├── ml_confusion_matrix.png# ML Confusion matrix plot
│   └── dl_training_curves.png # DL Accuracy & Loss training curves
│
├── training/
│   ├── train_ml.py            # ML training pipeline (TF-IDF + Logistic Regression)
│   └── train_lstm.py          # DL training pipeline (Embedding + Bi-LSTM)
│
├── chatbot/
│   ├── preprocessing.py       # Cleaning, contraction expansion, Kannada normalization
│   ├── predictor.py           # Unified inference engine & emergency check
│   └── response.py            # Knowledge base lookup & clinical disclaimers
│
├── database/
│   ├── database.py            # SQLite database schema, user auth, and logging
│   └── health_chatbot.db      # SQLite database file
│
├── templates/
│   ├── base.html              # Base HTML layout with emergency banner & navbar
│   ├── index.html             # Landing Home page
│   ├── chat.html              # Interactive Chatbot interface
│   ├── topics.html            # Searchable Health Topics directory
│   ├── about.html             # Project architecture & methodology
│   ├── faq.html               # Emergency guidance & FAQ
│   ├── login.html             # User login
│   ├── register.html          # User registration
│   └── admin.html             # Admin analytics dashboard
│
├── static/
│   ├── css/
│   │   └── style.css          # Design system & CSS
│   └── js/
│       ├── chatbot.js         # Chat UI logic, Speech Recognition & SpeechSynthesis
│       └── admin.js           # Admin Chart.js charts & stats updater
│
└── notebooks/
    ├── generate_notebooks.py  # Script to generate EDA & Comparison notebooks
    ├── data_analysis.ipynb    # Exploratory Data Analysis notebook
    └── model_comparison.ipynb# ML vs DL empirical comparison notebook
```

---

## 🚀 Quick Setup & Installation

### 1. Prerequisites
- Python 3.10+
- Modern Web Browser (Google Chrome, Edge, or Firefox)

### 2. Environment Setup
```bash
# Clone or navigate to the project directory
cd "e:\AI major Project\AI Health Chatbot"

# Activate virtual environment
.venv\Scripts\activate   # Windows
# or: source .venv/bin/activate  # macOS / Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Models (If retraining)
```bash
# Generate dataset
python data/generate_dataset.py

# Train ML Baseline (Logistic Regression)
python training/train_ml.py

# Train Deep Learning Architecture (Bi-LSTM)
python training/train_lstm.py
```

### 5. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
**http://127.0.0.1:5000**

---

## 📊 Empirical ML vs Deep Learning Comparison

Both models were trained and evaluated on an identical stratified split (80% Train, 10% Validation, 10% Test) of 5,843 questions across 18 intents:

| Metric | ML (TF-IDF + Logistic Regression) | DL (Embedding + Bi-LSTM) |
|---|---|---|
| **Test Accuracy** | 100.00% | > 98.5% |
| **Test Precision (Weighted)** | 100.00% | > 98.5% |
| **Test Recall (Weighted)** | 100.00% | > 98.5% |
| **Test F1-Score (Weighted)** | 100.00% | > 98.5% |
| **Training Duration** | ~1.3 seconds | ~45-60 seconds |
| **Inference Latency** | ~0.5 ms | ~15 ms |
| **Primary Strength** | Fast, transparent, zero hallucination | Sequential contextual understanding |

---

## 🛡️ Ethical Medical Disclaimer

> **IMPORTANT:** HealthAware AI is an informational and educational chatbot designed for public-health awareness. It **does not diagnose diseases**, recommend clinical prescriptions, or replace consultation with a licensed medical professional. In case of acute chest pain, breathing difficulty, or loss of consciousness, immediately call India's unified emergency number **112** or go to the nearest emergency department.
