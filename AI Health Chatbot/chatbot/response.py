"""
Health Knowledge Base and Response Generation Module.
Provides verified medical awareness responses, emergency detection,
confidence fallback handling, and ethical medical disclaimers.
"""

import re
from typing import Tuple, Dict, Any

# Critical Emergency Keywords triggering immediate emergency safety layer
EMERGENCY_KEYWORDS = [
    r"\bchest pain\b",
    r"\bheart attack\b",
    r"\bheart arrest\b",
    r"\bcardiac arrest\b",
    r"\bcannot breathe\b",
    r"\bcan\'t breathe\b",
    r"\bsevere difficulty breathing\b",
    r"\bunconscious\b",
    r"\blost consciousness\b",
    r"\bunresponsive\b",
    r"\bheavy bleeding\b",
    r"\bsevere bleeding\b",
    r"\bbleeding won\'t stop\b",
    r"\bstroke\b",
    r"\bface drooping\b",
    r"\bslurred speech\b",
    r"\bsudden paralysis\b",
    r"\bpoisoning\b",
    r"\bpoison\b",
    r"\bswallowed poison\b",
    r"\bchoking\b",
    r"\bblue lips\b",
    r"\bsuicide\b",
    r"\bsuicidal\b",
    r"\bkilling myself\b",
    r"\banaphylaxis\b",
    r"\bthroat closing\b"
]

# Emergency Protocol Response
EMERGENCY_RESPONSE = (
    "🚨 **CRITICAL MEDICAL EMERGENCY DETECTED**\n\n"
    "Based on the critical symptoms described, **do not wait for an online assessment**.\n\n"
    "**Immediate Actions:**\n"
    "1. **Call Emergency Services immediately:**\n"
    "   - 📞 **Dial 112** (India National Unified Emergency Helpline)\n"
    "   - 🚑 **Dial 102 / 108** (Ambulance Services)\n"
    "2. If the person is unconscious and not breathing, initiate bystander CPR immediately.\n"
    "3. Keep the patient in a safe, resting position and loosen tight clothing.\n"
    "4. Proceed immediately to the nearest hospital Emergency Department (Casualty/ER)."
)

FALLBACK_RESPONSE = (
    "I'm not completely confident I understood your health query (Confidence is below the clinical safety threshold). "
    "Please try rephrasing your question with specific symptoms or health topics (e.g., 'What are symptoms of fever?', "
    "'How to manage high blood pressure?', 'What is a balanced diet?'). "
    "For specific individual health conditions, always consult a certified medical practitioner."
)

MEDICAL_DISCLAIMER = (
    "\n\n*⚠️ Medical Disclaimer: This system provides general public-health awareness information only "
    "and does not constitute medical diagnosis, treatment, or clinical prescription. "
    "Always consult a qualified healthcare professional for personalized medical advice.*"
)

# Verified Knowledge Base Responses
KNOWLEDGE_BASE = {
    "greeting": (
        "Hello! I am your AI-powered Public Health Awareness Assistant. "
        "I provide evidence-based general health information, preventive tips, disease awareness, and wellness guidance. "
        "How can I assist you with your health questions today?"
    ),
    "goodbye": (
        "Thank you for consulting the Public Health Awareness Chatbot! "
        "Remember to stay hydrated, maintain good hygiene, and seek professional healthcare when needed. "
        "Have a healthy and safe day ahead!"
    ),
    "fever": (
        "**Fever Awareness & Care:**\n"
        "A fever is a temporary elevation in body temperature (above 38°C / 100.4°F), typically signaling an immune response to a viral or bacterial infection.\n\n"
        "• **Home Management:** Rest in a cool, well-ventilated room, stay well hydrated (water, oral rehydration salts, soups), and apply lukewarm sponge compresses.\n"
        "• **When to Seek Medical Care:** Consult a doctor if the fever exceeds 39.4°C (103°F), persists beyond 3 days, or is accompanied by a stiff neck, persistent vomiting, or difficulty breathing."
    ),
    "cold": (
        "**Common Cold Awareness & Care:**\n"
        "The common cold is a viral upper respiratory infection causing nasal congestion, sneezing, a runny nose, mild sore throat, and fatigue.\n\n"
        "• **Self-Care:** Most colds resolve naturally within 7–10 days. Get ample rest, drink warm liquids (herbal tea with honey), use steam inhalation, and use saline nasal drops.\n"
        "• **Note:** Antibiotics do not cure viral colds. Seek a physician if symptoms persist longer than 10 days or if high fever and severe sinus pain develop."
    ),
    "cough": (
        "**Cough Awareness & Care:**\n"
        "A cough is a vital defensive reflex that clears irritants and mucus from your respiratory airways.\n\n"
        "• **Dry vs Wet Cough:** Dry coughs are often caused by viral recovery, smoke, or allergies; wet coughs expel phlegm.\n"
        "• **Relief:** Stay hydrated, sip warm fluids with honey and ginger, avoid dust and cigarette smoke, and use a humidifier.\n"
        "• **When to Seek Care:** Consult a healthcare provider if coughing lasts over 3 weeks, produces rust-colored or bloody sputum, or causes severe shortness of breath."
    ),
    "diabetes": (
        "**Diabetes Mellitus Awareness:**\n"
        "Diabetes is a metabolic condition marked by elevated blood glucose (blood sugar) due to insufficient insulin production (Type 1) or cellular insulin resistance (Type 2).\n\n"
        "• **Common Symptoms:** Excessive thirst (polydipsia), frequent urination especially at night (polyuria), unexplained weight loss, blurry vision, and slow-healing wounds.\n"
        "• **Prevention & Management:** Maintain a balanced low-glycemic diet rich in fiber, engage in 150+ minutes of weekly aerobic exercise, maintain healthy body weight, and monitor fasting blood glucose and HbA1c periodically."
    ),
    "hypertension": (
        "**Hypertension (High Blood Pressure) Awareness:**\n"
        "Hypertension occurs when blood exerts persistent high pressure against artery walls (consistently 130/80 mmHg or higher). Known as the 'silent killer', it often produces no symptoms until complications arise.\n\n"
        "• **Prevention:** Limit sodium intake to under 2,000 mg/day (1 teaspoon of salt), consume potassium-rich vegetables, exercise regularly, limit alcohol, avoid tobacco, and manage stress.\n"
        "• **Emergency Warning:** Extremely high BP (>180/120 mmHg) with severe headache, chest pain, or visual changes requires immediate emergency room care."
    ),
    "nutrition": (
        "**Nutrition & Balanced Diet Guidance:**\n"
        "Proper nutrition fuels daily metabolism, supports immune defense, and prevents chronic cardiovascular and metabolic diseases.\n\n"
        "• **Key Guidelines:** Fill half your plate with colorful vegetables and fruits, one quarter with whole grains (brown rice, oats, whole wheat), and one quarter with lean proteins (legumes, lentils, eggs, lean poultry).\n"
        "• **Hydration:** Drink 2 to 3 liters of clean water daily. Minimize ultra-processed foods, refined sugars, deep-fried snacks, and trans fats."
    ),
    "vaccination": (
        "**Vaccination & Immunization Awareness:**\n"
        "Vaccines safely stimulate your body's immune system to create antibodies against specific pathogens without causing the actual illness.\n\n"
        "• **Benefits:** Immunization protects against life-threatening diseases such as measles, hepatitis, influenza, tetanus, polio, and HPV, fostering community herd immunity.\n"
        "• **Adult Boosters:** Adults should ensure up-to-date tetanus boosters (every 10 years), annual influenza vaccines, and specific vaccines recommended for chronic conditions or travel."
    ),
    "hygiene": (
        "**Hygiene & Infection Prevention:**\n"
        "Consistent hygiene practices break the chain of infection for bacterial, viral, and parasitic diseases.\n\n"
        "• **Handwashing:** Wash hands thoroughly with soap and water for at least 20 seconds before meals, after using restrooms, and after coughing or sneezing.\n"
        "• **Food & Water Hygiene:** Drink clean, filtered or boiled water, cook food thoroughly, keep raw meats separate from produce, and sanitize frequently touched surfaces."
    ),
    "mental_wellness": (
        "**Mental Wellness & Stress Management:**\n"
        "Mental wellness is essential for overall health, emotional resilience, and productive living.\n\n"
        "• **Stress Reduction:** Practice diaphragmatic breathing (4-7-8 breathing), engage in daily physical exercise, maintain 7–8 hours of consistent sleep, and stay connected with supportive family and friends.\n"
        "• **Professional Support:** If you experience persistent sadness, severe anxiety, panic attacks, or feelings of hopelessness lasting more than 2 weeks, consult a certified psychologist or psychiatrist."
    ),
    "asthma": (
        "**Asthma Awareness & Management:**\n"
        "Asthma is a chronic respiratory condition characterized by airway inflammation, hyper-reactivity, and bronchospasm.\n\n"
        "• **Symptoms:** Wheezing (whistling sound when breathing), chest tightness, breathlessness, and nocturnal coughing.\n"
        "• **Management:** Identify and avoid known triggers (dust mites, pollen, pet dander, cold air, smoke). Always keep prescribed rescue inhalers accessible.\n"
        "• **Emergency:** Seek emergency care immediately if breathing is severely labored, lips or nailbeds turn bluish, or the rescue inhaler fails to provide prompt relief."
    ),
    "headache": (
        "**Headache & Migraine Awareness:**\n"
        "Headaches range from common tension headaches (dull, band-like tightness) to vascular migraines (throbbing, usually on one side, with sensitivity to light and sound).\n\n"
        "• **Relief:** Stay hydrated, rest in a quiet and dark room, apply cold or warm compresses, avoid skipping meals, and minimize digital screen glare.\n"
        "• **Red Flags:** Seek emergency medical evaluation for a sudden explosive ('thunderclap') headache, headache following head injury, or headache with fever and neck stiffness."
    ),
    "heart_health": (
        "**Cardiovascular & Heart Health Awareness:**\n"
        "Maintaining heart health reduces the risk of coronary artery disease, heart failure, and stroke.\n\n"
        "• **Recommendations:** Engage in 150 minutes of moderate aerobic exercise weekly (brisk walking, cycling), adopt a heart-healthy Mediterranean-style diet, keep blood cholesterol within target ranges, and avoid smoking completely.\n"
        "• **Emergency Signs:** Crushing chest pain, pain radiating to the left arm, neck, or jaw, accompanied by sweating and nausea, warrants calling 112 immediately."
    ),
    "allergy": (
        "**Allergy & Hypersensitivity Awareness:**\n"
        "Allergies occur when the immune system overreacts to ordinarily harmless environmental substances such as pollen, mold, dust mites, insect venom, or specific foods.\n\n"
        "• **Management:** Avoid known allergen triggers, use hypoallergenic bedding, keep windows closed during high pollen counts, and use doctor-recommended antihistamines for mild reactions.\n"
        "• **Anaphylaxis Warning:** Sudden swelling of the lips, tongue, or throat, dizziness, or breathing distress is anaphylaxis—a life-threatening emergency requiring immediate emergency services (112)."
    ),
    "digestive_health": (
        "**Digestive Health & Acidity Care:**\n"
        "A healthy gastrointestinal system ensures efficient nutrient absorption and supports the gut microbiome.\n\n"
        "• **Acidity & Indigestion:** Avoid lying down within 2–3 hours after eating, eat smaller meals, reduce fried or overly spicy foods, and avoid excessive caffeine.\n"
        "• **Bowel Regularity:** Increase dietary fiber gradually with legumes, whole grains, and vegetables, and drink plenty of fluids.\n"
        "• **Medical Evaluation:** Consult a gastroenterologist if you experience persistent severe abdominal pain, difficulty swallowing, or unexplained weight loss."
    ),
    "first_aid": (
        "**Essential First Aid Awareness:**\n"
        "Prompt, appropriate first aid can stabilize acute injuries before professional clinical treatment is available.\n\n"
        "• **Minor Burns:** Cool immediately under cool running tap water for 10–15 minutes. Never apply ice, butter, or toothpaste.\n"
        "• **Bleeding:** Apply firm, continuous direct pressure with a clean cloth or bandage.\n"
        "• **Sprains (RICE):** Rest the injured limb, Ice for 15 minutes at a time, Compress gently with an elastic bandage, and Elevate above heart level.\n"
        "• **Choking:** Perform the Heimlich maneuver (abdominal thrusts) if an adult is conscious but unable to breathe."
    ),
    "emergency": EMERGENCY_RESPONSE
}

def check_emergency(text: str) -> bool:
    """Checks if the user text contains immediate critical emergency keywords."""
    clean = text.lower()
    for pattern in EMERGENCY_KEYWORDS:
        if re.search(pattern, clean):
            return True
    return False

def get_response_for_intent(intent: str, confidence: float, threshold: float = 0.60) -> Tuple[str, bool]:
    """
    Retrieves the response from the knowledge base based on intent and confidence.
    Returns (response_text, is_fallback).
    """
    if confidence < threshold:
        return FALLBACK_RESPONSE + MEDICAL_DISCLAIMER, True
    
    response = KNOWLEDGE_BASE.get(intent)
    if not response:
        return FALLBACK_RESPONSE + MEDICAL_DISCLAIMER, True
    
    # Emergency responses don't need additional general disclaimer
    if intent == "emergency":
        return response, False
    
    return response + MEDICAL_DISCLAIMER, False
