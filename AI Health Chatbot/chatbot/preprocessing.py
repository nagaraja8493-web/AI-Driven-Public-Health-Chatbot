"""
Text Preprocessing module for AI Public Health Chatbot.
Implements text cleaning, contraction expansion, Kannada transliteration mapping,
tokenization, and lemmatization.
"""

import re
import string

# Transliterated Kannada / Kannada health keywords mapping to English concepts
KANNADA_HEALTH_MAP = {
    r"\bjvara\b": "fever high temperature",
    r"\bjwara\b": "fever",
    r"\bseetha\b": "common cold runny nose",
    r"\bsheethe\b": "cold",
    r"\bkemmu\b": "cough",
    r"\bsakkare kayile\b": "diabetes high blood sugar",
    r"\bsakkare kayile laksanagalu\b": "diabetes symptoms",
    r"\brakta ottada\b": "hypertension high blood pressure",
    r"\baahara\b": "nutrition healthy diet",
    r"\blasike\b": "vaccination immunization",
    r"\bswachate\b": "hygiene cleanliness infection prevention",
    r"\bmanasika ottada\b": "mental wellness stress anxiety",
    r"\bdammu\b": "asthma breathing trouble",
    r"\btale novu\b": "headache migraine",
    r"\bhrudaya\b": "heart health cardiovascular",
    r"\bhotte novu\b": "digestive health stomach pain acidity",
    r"\bprathamachikitse\b": "first aid",
    r"\baakasmika\b": "emergency urgent help",
    r"\b112 karedi\b": "call 112 emergency helpline",
    r"\bdhanyavada\b": "thank you goodbye",
    r"\bnamaskara\b": "hello greetings",
    r"\bnamaste\b": "hello"
}

# English Contractions
CONTRACTIONS = {
    r"can\'t": "cannot",
    r"won\'t": "will not",
    r"n\'t": " not",
    r"\'re": " are",
    r"\'s": " is",
    r"\'d": " would",
    r"\'ll": " will",
    r"\'t": " not",
    r"\'ve": " have",
    r"\'m": " am"
}

def expand_contractions(text: str) -> str:
    """Expands common English contractions."""
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def normalize_multilingual(text: str) -> str:
    """Normalizes transliterated Kannada keywords to standard health vocabulary."""
    cleaned = text.lower()
    for pattern, replacement in KANNADA_HEALTH_MAP.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned

def clean_text(text: str) -> str:
    """
    Cleans raw input text:
    1. Lowercases
    2. Expands contractions
    3. Normalizes Kannada/transliterated terms
    4. Strips non-alphanumeric punctuation (except question marks for intent context)
    5. Normalizes whitespace
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    text = expand_contractions(text)
    text = normalize_multilingual(text)
    
    # Remove unwanted special characters, keeping alphanumeric and spaces
    text = re.sub(r"[^\w\s\?]", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def simple_tokenize(text: str) -> list:
    """Simple whitespace and punctuation-free tokenizer."""
    cleaned = clean_text(text)
    return [t for t in cleaned.split() if t]

def preprocess_pipeline(text: str) -> str:
    """End-to-end preprocessing pipeline for ML and DL inference."""
    cleaned = clean_text(text)
    return cleaned
