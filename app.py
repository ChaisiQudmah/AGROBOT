import streamlit as st
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------- FILE PATHS ----------
MODEL_PATH = "models/plant_disease_cnn.h5"
KNOWLEDGE_BASE_FILE = "knowledge_base.json"
USERS_FILE = "users.json"
CHAT_DB_FILE = "chat_database.json"

# ---------- LOAD USERS ----------
if os.path.exists(USERS_FILE):
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        USERS = json.load(f)
else:
    USERS = {}

# ---------- LOAD KNOWLEDGE BASE ----------
if os.path.exists(KNOWLEDGE_BASE_FILE):
    with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
else:
    knowledge_base = {
        "diseases": {},
        "general": {
            "EN": ["Sorry, I couldn't identify the disease. Please provide more details or an image."],
            "HI": ["माफ़ कीजिए, मैं रोग पहचान नहीं पाया। कृपया और जानकारी या छवि दें।"],
            "TE": ["క్షమించండి, నేను వ్యాధిని గుర్తించలేకపోయాను. దయచేసి మరిన్ని వివరాలు లేదా చిత్రం ఇవ్వండి."]
        }
    }

# ---------- LOAD CHAT DATABASE ----------
if os.path.exists(CHAT_DB_FILE):
    with open(CHAT_DB_FILE, "r", encoding="utf-8") as f:
        chat_db = json.load(f)
else:
    chat_db = {}

# ---------- LOAD IMAGE MODEL ----------
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=False)
        CLASS_NAMES = os.listdir("processed_dataset/train") if os.path.exists("processed_dataset/train") else []
    else:
        model = None
        CLASS_NAMES = []
except Exception:
    model = None
    CLASS_NAMES = []

# ---------- EMBEDDING MODEL ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2', device="cpu")

embedder = load_embedder()

# ---------- LANGUAGE DETECTION ----------
def detect_language(text):
    for char in text:
        if '\u0900' <= char <= '\u097F':  # Hindi
            return "HI"
        elif '\u0C00' <= char <= '\u0C7F':  # Telugu
            return "TE"
    return "EN"

# ---------- DISEASE MATCH ----------
def find_best_match(user_input, lang="EN"):
    user_vec = embedder.encode([user_input])
    best_match, best_score = None, -1
    for disease_key, data in knowledge_base.get("diseases", {}).items():
        samples = data["sample_inputs"].get(lang, [])
        if not samples:
            continue
        sample_vecs = embedder.encode(samples)
        sim = cosine_similarity(user_vec, sample_vecs).max()
        if sim > best_score:
            best_score = sim
            best_match = data
    if best_score > 0.55:
        return best_match
    return None

# ---------- CHAT RESPONSE ----------
def chat_response(user_input, lang="EN"):
    user_vec = embedder.encode([user_input])
    best_score, response = -1, None
    for intent, responses in chat_db.items():
        samples = responses.get(lang, [])
        if not samples:
            continue
        sample_vecs = embedder.encode(samples)
        sim = cosine_similarity(user_vec, sample_vecs).max()
        if sim > best_score:
            best_score = sim
            response = responses[lang][0]
    if best_score > 0.55 and response:
        return response
    else:
        return chat_db.get("intro", {}).get(lang, ["Hello! How can I help you?"])[0]

# ---------- SESSION STATE ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- LOGIN ----------
def login():
    st.title("🌱 AgroBot Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.role = USERS[username]["role"]
            st.session_state.username = username
            st.success(f"✅ Welcome {username}! Role: {USERS[username]['role']}")
        else:
            st.error("❌ Invalid credentials")

# ---------- FARMER PANEL ----------
def farmer_panel():
    st.subheader(f"Welcome {st.session_state.username}! Role: Farmer")

    # ----- Chatbot -----
    st.markdown("### 💬 AgroBot Chat")
    for msg in st.session_state.chat_history:
        if msg["sender"] == "user":
            st.markdown(f"👤 **You:** {msg['text']}")
        else:
            st.markdown(f"🤖 **AgroBot:** {msg['text']}")

    user_input = st.text_input("Type your message...", key="user_input")
    send_pressed = st.button("Send")

    if send_pressed and user_input.strip():
        if len(st.session_state.chat_history) == 0 or st.session_state.chat_history[-1]["text"] != user_input:
            st.session_state.chat_history.append({"sender": "user", "text": user_input})

            lang = detect_language(user_input)

            # Chat response
            chat_reply = chat_response(user_input, lang=lang)

            if any(greet in user_input.lower() for greet in ["hello", "hi", "hey", "bye", "నమస్తే", "హలో", "नमस्ते"]):
                response = chat_reply
            else:
                disease_match = find_best_match(user_input, lang=lang)
                if disease_match:
                    symptoms = ", ".join(disease_match['symptoms'].get(lang, []))
                    remedies_list = disease_match['remedies'].get(lang, [])
                    remedies = ", ".join(remedies_list) if remedies_list else "No specific remedies needed. 🌿"
                    response = f"Disease: {disease_match['disease'][lang]}\nSymptoms: {symptoms}\nRemedies: {remedies}"
                else:
                    response = chat_reply

            st.session_state.chat_history.append({"sender": "bot", "text": response})

    st.markdown("---")

    # ----- Image-based Disease Detection -----
    st.subheader("🌿 Image-based Disease Detection")
    uploaded = st.file_uploader("Upload plant image", type=["jpg", "png", "jpeg"], key="image_upload")

    if not model:
        st.warning("⚠️ Image prediction is disabled because the model could not be loaded.")
    elif uploaded:
        img = image.load_img(uploaded, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]
        pred_idx = np.argmax(preds)
        disease = CLASS_NAMES[pred_idx] if CLASS_NAMES else "Unknown Disease"
        confidence = preds[pred_idx] * 100

        kb_match = knowledge_base.get("diseases", {}).get(disease, None)
        if kb_match:
            remedies_list = kb_match.get("remedies", {}).get("EN", [])
            remedies = ", ".join(remedies_list) if remedies_list else None
            if remedies:
                st.success(f"Prediction: **{disease}** ({confidence:.2f}%)\nRemedies: {remedies}")
            else:
                st.success(f"Prediction: **{disease}** ({confidence:.2f}%)")
        else:
            st.success(f"Prediction: **{disease}** ({confidence:.2f}%)")

# ---------- ADMIN PANEL ----------
def admin_panel():
    st.subheader(f"Welcome {st.session_state.username}! Role: Admin")
    st.markdown("🛠 **Manage Knowledge Base**")

    # Load KB
    with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
        kb = json.load(f)

    diseases = list(kb.get("diseases", {}).keys())
    choice = st.selectbox("Select a disease to Edit/Delete or Add New", ["➕ Add New"] + diseases)

    if choice == "➕ Add New":
        new_key = st.text_input("Enter new disease key (e.g. Tomato_Blight)")
        if st.button("Create New Disease"):
            if new_key and new_key not in kb.get("diseases", {}):
                kb.setdefault("diseases", {})[new_key] = {
                    "disease": {"EN": "", "HI": "", "TE": ""},
                    "symptoms": {"EN": [], "HI": [], "TE": []},
                    "remedies": {"EN": [], "HI": [], "TE": []},
                    "sample_inputs": {"EN": [], "HI": [], "TE": []}
                }
                with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
                    json.dump(kb, f, ensure_ascii=False, indent=4)
                st.success(f"✅ New disease '{new_key}' added!")
                st.experimental_rerun()
    else:
        st.write(f"### Editing: {choice}")

        # --- Edit Fields ---
        for lang in ["EN", "HI", "TE"]:
            kb["diseases"][choice]["disease"][lang] = st.text_input(
                f"{lang} - Disease Name", value=kb["diseases"][choice]["disease"].get(lang, "")
            )
            kb["diseases"][choice]["symptoms"][lang] = [s.strip() for s in st.text_area(
                f"{lang} - Symptoms (comma separated)", 
                value=", ".join(kb["diseases"][choice]["symptoms"].get(lang, []))
            ).split(",") if s.strip()]
            kb["diseases"][choice]["remedies"][lang] = [s.strip() for s in st.text_area(
                f"{lang} - Remedies (comma separated)", 
                value=", ".join(kb["diseases"][choice]["remedies"].get(lang, []))
            ).split(",") if s.strip()]
            kb["diseases"][choice]["sample_inputs"][lang] = [s.strip() for s in st.text_area(
                f"{lang} - Sample Inputs (comma separated)", 
                value=", ".join(kb["diseases"][choice]["sample_inputs"].get(lang, []))
            ).split(",") if s.strip()]

        # --- Save Changes ---
        if st.button("💾 Save Changes"):
            with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
                json.dump(kb, f, ensure_ascii=False, indent=4)
            st.success("✅ Knowledge Base updated successfully!")

        # --- Delete Disease ---
        st.markdown("---")
        if st.button("🗑 Delete Disease"):
            kb["diseases"].pop(choice, None)  # Safely remove
            with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
                json.dump(kb, f, ensure_ascii=False, indent=4)
            st.success(f"❌ '{choice}' deleted!")
            st.experimental_rerun()  # Refresh page so selectbox updates

# ---------- MAIN ----------
if not st.session_state.logged_in:
    login()
else:
    if st.session_state.role == "Farmer":
        farmer_panel()
    elif st.session_state.role == "Admin":
        admin_panel()
