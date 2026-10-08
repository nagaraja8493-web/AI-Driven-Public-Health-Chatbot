"""
Dataset Generator for Public Health Awareness Chatbot
Generates data/intents.csv containing diverse questions and verified responses across 18 intents.
Includes both English and Kannada/Kannada-transliterated health questions.
"""

import os
import random
import pandas as pd

INTENT_DATA = {
    "greeting": {
        "response": "Hello! I am your Public Health Awareness Assistant. I provide evidence-based general health information, preventive tips, and wellness guidance. How can I assist you with your health questions today?",
        "templates": [
            "Hello", "Hi", "Hey there", "Good morning", "Good afternoon", "Good evening",
            "Greetings", "Hi bot", "Hello assistant", "Hey chatbot", "Hi there, can you help me?",
            "Are you there?", "Start conversation", "Hello health bot", "Hey, I have a question",
            "Can I ask a health question?", "Namaskara", "Namaste", "Hello doctor assistant",
            "Howdy", "Hi, I need some health info", "Hello there, need advice",
            "Hey, can we chat about health?", "Greetings health assistant",
            "Hi health advisor", "Hello friendly bot", "Good day to you",
            "Namaskara, can you assist me?", "Hey bot, quick question"
        ],
        "expansions": [
            "I need some quick guidance", "are you available right now?", "can I speak with you?",
            "I want to know about healthy living", "hope you are doing well",
            "ready to assist me today?", "looking for public health advice",
            "need some medical awareness info", "glad to talk to you"
        ]
    },
    "goodbye": {
        "response": "Thank you for consulting the Public Health Awareness Chatbot! Remember to prioritize healthy habits and consult a certified medical practitioner for personalized healthcare. Stay safe and healthy!",
        "templates": [
            "Goodbye", "Bye", "See you later", "Thanks", "Thank you very much",
            "Thanks for your help", "Bye bye", "Have a great day", "Thank you doctor bot",
            "That answered my question, thank you", "Talk to you later", "Signing off",
            "Dhanyavada", "Thank you so much", "Appreciate your advice", "Bye for now",
            "Catch you later", "Thanks a lot for the information", "I am done for today, thanks",
            "Okay bye", "Thank you, that helps a lot", "Good night and thank you"
        ],
        "expansions": [
            "have a good day ahead", "take care", "will check back later",
            "much appreciated", "that was very informative", "stay safe"
        ]
    },
    "fever": {
        "response": "A fever is a temporary rise in body temperature (typically above 38°C or 100.4°F), most often caused by a viral or bacterial infection. Stay well hydrated, rest adequately, and use cool compresses. If fever exceeds 39.4°C (103°F), lasts more than 3 days, or is accompanied by stiff neck, breathing difficulties, or severe rash, seek clinical care promptly.",
        "templates": [
            "What is fever?", "What are the symptoms of fever?", "How do I know if I have a fever?",
            "What causes high body temperature?", "How to treat fever at home?",
            "What to do when body temperature rises?", "My body feels very hot and I am shivering",
            "What temperature is considered a fever?", "How to reduce fever naturally?",
            "Can viral infection cause fever?", "Why do I feel chills and shivering with fever?",
            "How long does a viral fever usually last?", "Is 102 degrees fever dangerous?",
            "Best home remedies for fever", "When should I see a doctor for high fever?",
            "Nanage jvara ide, en madbeku?", "Fever and headache management",
            "What medicines or fluids should I take during fever?", "How to manage high temperature?",
            "Can dehydration cause mild fever?", "What are the warning signs of high fever in adults?",
            "How can I measure body temperature accurately?", "What causes recurrent fever episodes?"
        ],
        "expansions": [
            "in adults and elderly", "during monsoon or winter season", "with body ache and shivering",
            "what home care steps are recommended?", "and what complications should I watch out for?",
            "when is it considered an emergency?", "how much water or fluids should I drink?",
            "and feeling weak and dizzy"
        ]
    },
    "cold": {
        "response": "The common cold is an upper respiratory viral illness marked by nasal congestion, sneezing, a runny nose, mild fatigue, and sore throat. Most colds resolve within 7 to 10 days. Rest, saline nasal sprays, warm honey water, and steam inhalation help relieve symptoms. Consult a physician if symptoms persist past 10 days or worsen significantly.",
        "templates": [
            "What are common cold symptoms?", "How to treat a common cold?", "How can I manage runny nose and sneezing?",
            "What is the difference between a cold and the flu?", "How long does a common cold last?",
            "Best remedies for blocked nose and congestion", "Why do I catch cold frequently?",
            "Can cold weather cause catching a cold?", "How to prevent catching a cold from others?",
            "Steam inhalation for cold relief", "Is sore throat a symptom of common cold?",
            "How to stop sneezing and nose dripping?", "Nanage seetha agide, en parihara?",
            "What home remedies cure cold quickly?", "How can I soothe an irritated throat from cold?",
            "Are antibiotics needed for a common cold?", "How does the common cold virus spread?",
            "What vitamins help relieve common cold?", "Tips for sleeping with a stuffed up nose"
        ],
        "expansions": [
            "especially at night", "without taking heavy medications", "using safe home practices",
            "and preventing it from spreading to family members", "when should I consult an ENT specialist?",
            "with continuous watery eyes and sneezing", "during changing seasons"
        ]
    },
    "cough": {
        "response": "Coughing is your respiratory system's protective reflex to expel irritants, allergens, or mucus. Dry coughs often follow viral infections or allergies, while wet coughs clear phlegm. Stay hydrated, sip warm herbal tea with honey, and avoid smoke. Consult a healthcare provider if your cough lasts more than 3 weeks, causes chest pain, or produces blood.",
        "templates": [
            "What causes coughing?", "How to treat persistent dry cough?", "What is the difference between dry cough and wet cough?",
            "Home remedies for severe night cough", "Why do I keep coughing all the time?",
            "How to clear phlegm and mucus from chest?", "Can pollution trigger chronic cough?",
            "When is coughing a sign of serious lung illness?", "How can warm water and honey help cough?",
            "Nanage kemmu ide, en madodu?", "Best drinks to soothe an irritated throat and cough",
            "What causes cough that worsens when lying down?", "Can acid reflux cause chronic coughing?",
            "How to stop ticklish throat cough?", "Is coughing blood a medical emergency?",
            "Cough relief tips for cold weather", "What lifestyle changes reduce respiratory cough?"
        ],
        "expansions": [
            "that doesn't go away after two weeks", "with thick yellow or green sputum",
            "especially during nighttime sleep", "what home treatments are scientifically safe?",
            "and when is chest X-ray or doctor visit advised?", "associated with allergic sensitivity"
        ]
    },
    "diabetes": {
        "response": "Diabetes mellitus is a metabolic condition characterized by high blood glucose levels resulting from insufficient insulin production or ineffective insulin action. Hallmark signs include excessive thirst, frequent urination, constant hunger, blurred vision, and unexplained weight loss. Manage it with regular aerobic exercise, a low-glycemic high-fiber diet, weight management, and medical follow-up.",
        "templates": [
            "What is diabetes?", "What are the common symptoms of diabetes?", "What is the difference between Type 1 and Type 2 diabetes?",
            "How is high blood sugar diagnosed?", "What is normal fasting blood sugar level?",
            "What is HbA1c test and what does it mean?", "How can I prevent Type 2 diabetes?",
            "Can eating too much sugar directly cause diabetes?", "What foods should diabetics avoid?",
            "What are the early warning signs of diabetes?", "Sakkare kayile laksanagalu enu?",
            "How does exercise help manage blood glucose?", "What causes sudden hypoglycemia or low sugar?",
            "What complications arise from uncontrolled diabetes?", "Is diabetes completely curable or manageable?",
            "How often should a diabetic check blood glucose?", "Can stress elevate blood glucose levels?",
            "Foot care guidelines for diabetes patients", "Best low glycemic foods for diabetic diet"
        ],
        "expansions": [
            "in middle aged and older adults", "and what lifestyle changes can reverse prediabetes?",
            "to prevent cardiovascular and kidney complications", "with frequent urination at night and excessive thirst",
            "what should be the ideal fasting and post-prandial range?", "how to manage it through natural diet and walking"
        ]
    },
    "hypertension": {
        "response": "Hypertension (high blood pressure) occurs when blood pushes against artery walls with persistently high force (130/80 mmHg or higher). Often asymptomatic ('the silent killer'), untreated hypertension can damage the heart, brain, and kidneys. Prevention includes reducing dietary sodium (<2g/day), regular aerobic exercise, limiting alcohol, stress management, and maintaining healthy weight.",
        "templates": [
            "What is hypertension?", "What is high blood pressure?", "What are the symptoms of high blood pressure?",
            "What is considered normal blood pressure reading?", "What causes sudden rise in blood pressure?",
            "How can I reduce blood pressure naturally?", "Why is hypertension called a silent killer?",
            "What foods help lower blood pressure?", "How does salt intake affect hypertension?",
            "Can stress and anxiety cause chronic high BP?", "Rakta ottada samasyege parihara enu?",
            "What is systolic vs diastolic blood pressure?", "What are the dangers of untreated hypertension?",
            "How much exercise is recommended for high BP?", "Can meditation and yoga lower blood pressure?",
            "What are hypertensive crisis warning signs?", "When should I monitor my blood pressure at home?"
        ],
        "expansions": [
            "without relying solely on pills", "and what dietary changes like DASH diet are effective?",
            "how to recognize early warning signs before stroke", "what is the safe limit of daily sodium consumption?",
            "and its relation to heart attack risk"
        ]
    },
    "nutrition": {
        "response": "A wholesome, balanced diet provides vital macro- and micronutrients to fuel body metabolism, fortify immunity, and prevent chronic illness. Emphasize whole grains, diverse colorful vegetables, fresh fruits, legumes, healthy unsaturated fats, and adequate hydration (2 to 3 liters daily). Minimize ultra-processed foods, refined sugars, and trans fats.",
        "templates": [
            "What is a healthy balanced diet?", "What nutrients does the human body need daily?",
            "How much water should I drink every day?", "What are the best immunity boosting foods?",
            "What is the difference between good fats and bad fats?", "How to plan a balanced vegetarian diet?",
            "Why is dietary fiber important for health?", "What foods are rich in protein for daily energy?",
            "How does excessive sugar consumption harm the body?", "What are the symptoms of vitamin D deficiency?",
            "Uttama aahara kramavannu vivarisi?", "Healthy snacking options for weight control",
            "How to calculate daily caloric requirements?", "Why are micronutrients like iron and zinc important?",
            "Is intermittent fasting safe for healthy adults?", "How does processed food impact gut microbiome?"
        ],
        "expansions": [
            "to build natural immunity and stamina", "for sustained physical energy and focus",
            "while maintaining optimal body mass index", "what foods should be included in daily meal planning?",
            "and avoiding nutritional deficiencies"
        ]
    },
    "vaccination": {
        "response": "Vaccines train the immune system to recognize and neutralize specific infectious viruses and bacteria without causing the disease itself. Immunization prevents severe illnesses like measles, hepatitis, influenza, tetanus, polio, and COVID-19. Keeping vaccines updated safeguards both individuals and community herd immunity.",
        "templates": [
            "Why are vaccines important?", "How do vaccines work inside the human body?",
            "What is herd immunity?", "What vaccines do adults need to take?",
            "Are vaccines safe and scientifically tested?", "What are common mild side effects after vaccination?",
            "Why is the annual flu shot recommended?", "What is the tetanus booster schedule?",
            "Lasikeya mukhya prayojanagalu enu?", "How do mRNA vaccines differ from traditional vaccines?",
            "Can vaccines cause the illness they protect against?", "Why should travelers receive specific immunizations?",
            "What vaccines protect against cervical cancer?", "How does childhood immunization save lives?",
            "What should I do if I miss a scheduled vaccine dose?"
        ],
        "expansions": [
            "according to World Health Organization guidelines", "for long term disease eradication",
            "and debunking common myths about vaccine safety", "for children and immunocompromised individuals",
            "and what precautions to take before receiving a jab"
        ]
    },
    "hygiene": {
        "response": "Good personal and environmental hygiene is the first line of defense against infectious diseases. Wash hands with soap and clean water for at least 20 seconds, especially before meals and after using restrooms. Practice safe food handling, maintain clean drinking water, ensure proper ventilation, and sanitize frequently handled surfaces.",
        "templates": [
            "How can I prevent infections through hygiene?", "What is the proper handwashing technique?",
            "How long should I wash my hands with soap?", "Why is sanitizing frequently touched objects important?",
            "What are best practices for food hygiene?", "How to purify drinking water at home safely?",
            "How does personal hygiene prevent skin infections?", "Swachate mattu aarogya kaapadukolluvudu hege?",
            "What is respiratory etiquette when coughing or sneezing?", "How does oral dental hygiene affect overall health?",
            "Tips for preventing foodborne bacterial illness", "Why should raw meat and vegetables be kept separate?",
            "How often should toothbrushes and towels be replaced?", "What hygiene habits protect against waterborne diseases?"
        ],
        "expansions": [
            "in everyday household and workplace environments", "during outbreaks and seasonal flu spreads",
            "to safeguard family members and children", "what guidelines does public health emphasize?",
            "to prevent cross contamination of food and surfaces"
        ]
    },
    "mental_wellness": {
        "response": "Mental wellness encompasses emotional, psychological, and social well-being. Chronic stress, anxiety, and depression can manifest in sleep disturbances, fatigue, and difficulty concentrating. Promote mental well-being with consistent sleep routines, mindfulness or deep breathing, regular physical exercise, and open social connection. Seek certified counseling if feelings become overwhelming.",
        "templates": [
            "How can I manage stress effectively?", "What are the common signs of clinical depression?",
            "How to overcome daily anxiety and overthinking?", "What relaxation techniques help calm the mind?",
            "How does regular sleep impact mental health?", "Manasika ottada nivaranege kramagalu enu?",
            "What is burnout and how do I recover from it?", "How does exercise boost mental well-being and mood?",
            "Tips for practicing mindfulness and meditation", "How can I support a friend going through depression?",
            "What are the physical symptoms of chronic stress?", "When should someone seek therapy or psychiatric help?",
            "How to cope with panic attacks and sudden anxiety?", "Ways to improve work-life balance and mental peace?",
            "How does digital detox benefit psychological health?"
        ],
        "expansions": [
            "without relying on sedative drugs", "in stressful academic or professional settings",
            "to build emotional resilience and calm", "and recognizing when to talk to a licensed therapist",
            "for better sleep hygiene and cognitive clarity"
        ]
    },
    "asthma": {
        "response": "Asthma is a chronic inflammatory disorder of the airways causing recurring episodes of wheezing, shortness of breath, chest tightness, and coughing. Common triggers include dust mites, animal dander, pollen, cold air, smoke, and respiratory infections. Keep doctor-prescribed rescue and controller inhalers on hand and develop an Asthma Action Plan.",
        "templates": [
            "What is asthma?", "What are the primary triggers of asthma attacks?",
            "How do inhalers work for asthma management?", "What are the early warning signs of an asthma flare-up?",
            "Can exercise trigger asthma symptoms?", "How to create an asthma-friendly home environment?",
            "What is the difference between controller and rescue inhalers?", "Dammu mattu usirata samasye laksana enu?",
            "How does air pollution aggravate bronchial asthma?", "What should you do if someone has an asthma attack?",
            "Can allergies develop into chronic asthma?", "How is peak flow meter used to monitor lung capacity?",
            "Tips for managing cold weather induced asthma", "When is an asthma attack considered life-threatening?"
        ],
        "expansions": [
            "especially in urban dusty environments", "and what immediate steps to take during shortness of breath",
            "for children and adults living with respiratory sensitivity", "how to properly use a spacer with an inhaler",
            "and avoiding pollen, smoke, and temperature extremes"
        ]
    },
    "headache": {
        "response": "Headaches vary from tension headaches (muscle contraction in scalp and neck) to vascular migraines (throbbing, often unilateral, accompanied by light/sound sensitivity or nausea). Relief involves hydration, resting in a quiet dark room, stress relief, and cold/warm compresses. Seek immediate emergency care for sudden 'thunderclap' headaches or headaches accompanied by confusion, stiff neck, or numbness.",
        "templates": [
            "What causes frequent headaches?", "What is the difference between tension headache and migraine?",
            "How to relieve migraine pain naturally?", "Can dehydration and lack of sleep cause headaches?",
            "What are migraine triggers to avoid?", "Tale novu parihara hege?",
            "Why do I get headaches after looking at computer screens?", "What causes cluster headaches?",
            "Home remedies for sinus headache and pressure", "When is a headache a sign of a serious brain emergency?",
            "Can skipping meals trigger throbbing head pain?", "How does caffeine affect headaches?",
            "Tips for neck and shoulder relaxation to cure tension headaches", "What is an aura in migraine?"
        ],
        "expansions": [
            "that occur almost daily", "with sensitivity to bright lights and loud sounds",
            "and what red-flag symptoms require immediate MRI or ER visit?", "using simple ergonomic and hydration remedies",
            "after long working hours on digital screens"
        ]
    },
    "heart_health": {
        "response": "Cardiovascular wellness requires maintaining healthy blood vessels and heart muscle function to avoid coronary artery disease and heart attacks. Protect your heart with 150 minutes of weekly aerobic exercise, a Mediterranean or DASH diet low in trans fats, maintaining normal blood pressure, quitting smoking, and keeping LDL cholesterol low. Dial 112 immediately for crushing chest pain, cold sweats, or arm radiation.",
        "templates": [
            "How to keep your heart healthy?", "What are the early warning signs of heart disease?",
            "What is the difference between good cholesterol and bad cholesterol?", "How can I lower LDL cholesterol naturally?",
            "What foods are best for heart health?", "Hrudaya aarogya kaapadukolluvudu hege?",
            "How does aerobic exercise strengthen the heart muscle?", "What causes atherosclerosis or blocked arteries?",
            "Can chronic stress lead to heart attacks?", "What is a healthy resting heart rate for adults?",
            "How does smoking damage heart vessels?", "What are the symptoms of angina pectoris?",
            "Tips for preventing coronary artery disease", "How does diabetes increase the risk of heart disease?"
        ],
        "expansions": [
            "for people above 40 years of age", "with family history of cardiac problems",
            "and what preventive screenings like ECG or lipid profile to get", "through sustainable daily dietary and walking routines",
            "and recognizing non-typical heart symptoms in women"
        ]
    },
    "allergy": {
        "response": "Allergies represent an exaggerated immune response to harmless foreign substances (allergens) such as pollen, dust mites, insect venom, animal fur, or specific foods (peanuts, shellfish, dairy). Symptoms include sneezing, itchy watery eyes, hives, or swelling. Mild allergies respond to over-the-counter antihistamines. Anaphylaxis is a life-threatening medical emergency requiring immediate epinephrine and hospital transport.",
        "templates": [
            "What causes allergic reactions?", "What are common signs of seasonal pollen allergy?",
            "How do antihistamines relieve allergy symptoms?", "What are common food allergens to watch out for?",
            "What is allergic rhinitis and how is it treated?", "Allergy laksanagalu mattu parihara enu?",
            "How can I test what I am allergic to?", "What is anaphylaxis and what are its symptoms?",
            "How to manage skin hives and allergic itching?", "Can dust mite allergies cause nighttime congestion?",
            "What precautions should people with food allergies take?", "Difference between food allergy and food intolerance",
            "How to reduce pet dander allergy at home?", "When does an allergic reaction require emergency adrenaline?"
        ],
        "expansions": [
            "with sudden skin rashes, itching, and redness", "during spring and autumn seasonal shifts",
            "and how to prevent accidental exposure to trigger foods", "what tests like skin prick or IgE are available?",
            "and recognizing severe throat swelling and breathing trouble"
        ]
    },
    "digestive_health": {
        "response": "Gastrointestinal health depends on balanced digestion, nutrient absorption, and a healthy gut microbiome. Common issues like acid reflux (GERD), bloating, constipation, and gastritis improve with smaller frequent meals, chewing thoroughly, high-fiber foods, adequate hydration, probiotics (such as yogurt), and avoiding late-night heavy meals. Consult a physician for persistent sharp pain or blood in stool.",
        "templates": [
            "How to improve digestive health?", "What causes acid reflux and heartburn?",
            "Home remedies for stomach acidity and gas", "How to relieve constipation naturally?",
            "What foods promote healthy gut bacteria?", "Hotte novu mattu acidity samasye hege nivarisuwudu?",
            "Why do I feel bloated after eating?", "What is gastritis and what causes it?",
            "How much dietary fiber is needed for smooth digestion?", "Can drinking warm water aid digestion?",
            "What causes sudden stomach cramps and diarrhea?", "How does stress affect the digestive system and IBS?",
            "Tips for preventing indigestion and nausea", "When is severe stomach pain a surgical emergency?"
        ],
        "expansions": [
            "after eating spicy, oily, or fried foods", "to maintain regular and comfortable bowel movements",
            "through natural probiotic foods and gentle movement", "and warning signs of ulcers or appendicitis",
            "without relying continuously on antacid medications"
        ]
    },
    "first_aid": {
        "response": "Basic first aid provides immediate stabilization for injuries before professional medical care arrives. For minor burns, cool under clean running water for 10-15 minutes (never use ice or toothpaste). For bleeding, apply direct continuous pressure with a clean cloth. For sprains, follow RICE (Rest, Ice, Compression, Elevation). Seek immediate emergency medical care for fractures, deep wounds, or uncontrolled bleeding.",
        "templates": [
            "What are basic first aid steps for minor burns?", "How to treat a bleeding wound or deep cut?",
            "What is the RICE protocol for sprains and strains?", "What essential items should be in a home first aid kit?",
            "How to treat a nosebleed correctly?", "Prathamachikitse kramagalu yavuvu?",
            "What to do if someone faints or loses consciousness?", "How to treat a bee or insect sting at home?",
            "What is first aid for suspected bone fractures?", "How to perform the Heimlich maneuver for choking?",
            "What should you not do when treating a burn?", "How to clean and disinfect an abrasion or scrape?",
            "First aid response for accidental chemical splash in eye", "How to bandage a twisted ankle safely?"
        ],
        "expansions": [
            "until an ambulance or doctor is available", "in home, school, and road accident scenarios",
            "to prevent infection and promote quick healing", "without causing further tissue trauma",
            "and when emergency transportation to a hospital is required"
        ]
    },
    "emergency": {
        "response": "CRITICAL EMERGENCY ALERT: If you or someone nearby is experiencing acute chest pain radiating to the jaw/arm, severe difficulty breathing, sudden face drooping or speech difficulty (stroke), heavy uncontrollable bleeding, sudden collapse, or suicidal thoughts, DO NOT WAIT. In India, IMMEDIATELY call 112 (National Emergency Helpline) or 102/108 (Ambulance Service), or go to the nearest emergency department right now.",
        "templates": [
            "I need emergency help right now", "Emergency medical situation", "Call an ambulance immediately",
            "Severe chest pain and can't breathe", "Heart attack symptoms emergency",
            "Someone collapsed and is unconscious", "Patient is not breathing, need urgent help",
            "Heavy bleeding won't stop, emergency", "Signs of stroke, face drooping and slurred speech",
            "Aakasmika emergency, 112 karedi madbeka?", "Poisoning emergency, swallowed toxic substance",
            "Severe head injury and bleeding", "Sudden paralysis on one side of the body",
            "Choking and unable to inhale air", "Extreme shortness of breath and blue lips",
            "Severe allergic shock, throat closing", "Urgent hospital helpline number",
            "What is India national emergency number?", "I need an urgent doctor right now"
        ],
        "expansions": [
            "please tell me the immediate emergency hotline", "what should I do while waiting for the ambulance?",
            "this is an urgent life-threatening situation", "need fast emergency intervention right away",
            "vital signs are deteriorating rapidly"
        ]
    }
}

def generate_dataset(target_total=1800, output_path="data/intents.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    
    # Prefix / question wrapper variants
    prefixes = [
        "", "Can you tell me, ", "Please explain, ", "I want to ask, ", "Tell me, ",
        "Could you clarify, ", "Doctor, ", "Bot, ", "Hello, ", "Kindly explain, ",
        "Need advice on: ", "Question regarding: ", "How do I understand: ", ""
    ]
    
    random.seed(42)
    
    for intent, data in INTENT_DATA.items():
        base_templates = data["templates"]
        expansions = data.get("expansions", [""])
        response = data["response"]
        
        seen = set()
        
        # 1. Add direct templates
        for t in base_templates:
            clean_t = t.strip()
            if clean_t and clean_t.lower() not in seen:
                seen.add(clean_t.lower())
                rows.append({"text": clean_t, "intent": intent, "response": response})
        
        # 2. Add prefix combinations
        for t in base_templates:
            for p in prefixes:
                if not p:
                    continue
                combined = f"{p}{t[0].lower()}{t[1:]}" if len(t) > 1 else f"{p}{t}"
                combined = combined.strip()
                if combined.lower() not in seen:
                    seen.add(combined.lower())
                    rows.append({"text": combined, "intent": intent, "response": response})
        
        # 3. Add expansions
        for t in base_templates:
            for exp in expansions:
                if not exp:
                    continue
                # If t ends with '?', replace or append
                if t.endswith("?"):
                    combined = f"{t[:-1]} {exp}?"
                else:
                    combined = f"{t} {exp}"
                combined = combined.strip()
                if combined.lower() not in seen:
                    seen.add(combined.lower())
                    rows.append({"text": combined, "intent": intent, "response": response})

    df = pd.DataFrame(rows)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Generated dataset with {len(df)} samples across {len(INTENT_DATA)} intents saved to {output_path}.")
    print("\nSamples per intent:")
    print(df["intent"].value_counts())
    return df

if __name__ == "__main__":
    generate_dataset()
