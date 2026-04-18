import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import json
import requests
from groq import Groq
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Based FIR & BNS Prediction System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL LOADING  (Phase 2 pkl files)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    tfidf = joblib.load("ai_module/models/tfidf_vectorizer.pkl")
    model = joblib.load("ai_module/models/bns_model.pkl")
    lr_model = joblib.load("ai_module/models/bns_lr_model.pkl")
    le = joblib.load("ai_module/models/label_encoder.pkl")

    # Compatibility patch
    for m in [model, lr_model]:
        if hasattr(m, "__class__") and m.__class__.__name__ == "LogisticRegression":
            if not hasattr(m, "multi_class"):
                m.multi_class = "auto"

    return tfidf, model, lr_model, le

@st.cache_data
def load_bns_sections():
    df = pd.read_csv("ai_module/datasets/bns_sections.csv")
    # Build lookup: section_number → {name, description, chapter}
    lookup = {}
    
    for _, row in df.iterrows():
        sec = int(row["Section"])
        lookup[sec] = {
            "chapter"     : int(row["Chapter"]),
            "chapter_name": str(row["Chapter_name"]),
            "section_name": str(row["Section _name"]),
            "description" : str(row["Description"])
        }
    return lookup

@st.cache_data
def load_fir_dataset():
    return pd.read_csv("ai_module/datasets/fir_dataset_with_id.csv")

tfidf, model, lr_model, le = load_models()
bns_lookup  = load_bns_sections()
fir_dataset = load_fir_dataset()

# ─────────────────────────────────────────────────────────────────────────────
# 3. UTILITY FUNCTIONS  (Phase 3 requirements)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Normalise raw FIR complaint text for model input."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_phone(phone: str) -> bool:
    """
    Validate Indian mobile numbers.
    Accepts: 10-digit numbers starting with 6-9,
             optionally prefixed with +91 or 0.
    """
    phone = phone.strip().replace(" ", "").replace("-", "")
    pattern = r"^(?:\+91|91|0)?[6-9]\d{9}$"
    return bool(re.fullmatch(pattern, phone))


def get_location_suggestions(query: str, limit: int = 5) -> list[dict]:
    """
    Fetch location autocomplete suggestions via Nominatim (OpenStreetMap).
    No API key required. Biased toward India.
    Returns list of dicts with keys: display_name, lat, lon.
    """
    if not query or len(query) < 3:
        return []
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q"              : query,
            "format"         : "json",
            "limit"          : limit,
            "countrycodes"   : "in",   # restrict to India
            "addressdetails" : 0,
        }
        headers = {"User-Agent": "AI-FIR-BNS-System/2.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        resp.raise_for_status()
        results = resp.json()
        return [
            {
                "display_name": r.get("display_name", ""),
                "lat"         : r.get("lat", ""),
                "lon"         : r.get("lon", "")
            }
            for r in results
        ]
    except Exception:
        return []


def predict_bns(text: str, top_k: int = 5) -> list[dict]:
    """
    Predict BNS sections for a given FIR complaint text.
    Returns top-K list of {section, confidence, section_name, chapter_name}.
    """
    cleaned = preprocess_text(text)
    if not cleaned:
        return []

    vec = tfidf.transform([cleaned])

    # Use Logistic Regression model for probabilities
    try:
        probs = lr_model.predict_proba(vec)[0]
    except Exception:
        # fallback patch for sklearn version mismatch
        if not hasattr(lr_model, "multi_class"):
            lr_model.multi_class = "auto"
        probs = lr_model.predict_proba(vec)[0]

    top_idx = np.argsort(probs)[::-1][:top_k]

    results = []
    for idx in top_idx:
        section_num = int(le.inverse_transform([idx])[0])
        info = bns_lookup.get(section_num, {})

        results.append({
            "section": section_num,
            "confidence": round(float(probs[idx]) * 100, 2),
            "section_name": info.get("section_name", "—"),
            "chapter_name": info.get("chapter_name", "—"),
            "description": info.get("description", "—")[:300] + "..."
        })

    return results


def get_similar_firs(query_text: str, top_n: int = 5) -> pd.DataFrame:
    """
    Find the most similar past FIRs using cosine similarity on TF-IDF vectors.
    Returns a DataFrame of top_n similar FIRs with similarity scores.
    """
    cleaned   = preprocess_text(query_text)
    query_vec = tfidf.transform([cleaned])

    corpus_texts = fir_dataset["fir_text"].apply(preprocess_text).tolist()
    corpus_vecs  = tfidf.transform(corpus_texts)

    sims     = cosine_similarity(query_vec, corpus_vecs)[0]
    top_idx  = np.argsort(sims)[::-1][:top_n]

    results = fir_dataset.iloc[top_idx].copy()
    results["similarity"] = (sims[top_idx] * 100).round(2)

    # Enrich with BNS section names
    results["section_name"] = results["bns_section"].apply(
        lambda s: bns_lookup.get(int(s), {}).get("section_name", "—")
    )
    return results[["fir_id", "fir_text", "bns_section", "section_name", "similarity"]]


# ─────────────────────────────────────────────────────────────────────────────
# 3b. CHATBOT FUNCTION (Groq API)
# ─────────────────────────────────────────────────────────────────────────────

def call_groq(user_message: str, history: list) -> str:
    """
    Send a message to Groq LLM (Llama 3.3 70B) with BNS-focused system prompt.
    history: list of {role, content} dicts (last 10 messages used for context).
    Returns the assistant reply as a string.
    """
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if not api_key:
            return (
                "⚠️ Groq API key not configured. "
                "Please add GROQ_API_KEY to `.streamlit/secrets.toml` and restart the app."
            )

        client = Groq(api_key=api_key)

        system_prompt = (
            "You are a legal assistant specializing in the Bharatiya Nyaya Sanhita (BNS) 2023 — "
            "India's new criminal code that replaced the Indian Penal Code (IPC). "
            "Your role is to help police officers and complainants understand BNS sections, "
            "the FIR filing process, legal procedures, and criminal law concepts. "
            "Always be concise, factual, and use simple language. "
            "When referencing sections, always say 'BNS Section X' (not IPC). "
            "Do not provide personal legal advice or guarantee outcomes. "
            "If a question is outside criminal law, politely say so."
        )

        messages = [{"role": "system", "content": system_prompt}]
        # Include last 10 turns for context window efficiency
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=600,
            temperature=0.4,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error contacting Groq API: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. SESSION STATE & LOGIN  (Phase 4 RBAC)
# ─────────────────────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in   = False
    st.session_state.role        = None
    st.session_state.username    = ""
    st.session_state.fir_store   = []   # in-memory FIR list for demo
    st.session_state.chat_history = []  # chatbot conversation history


def login(username: str, password: str, role: str) -> bool:
    """
    Demo login — replace with a real DB check in production.
    Any non-empty credentials work for demo purposes.
    """
    if username.strip() and password.strip():
        st.session_state.logged_in = True
        st.session_state.role      = role
        st.session_state.username  = username.strip()
        return True
    return False


def logout():
    st.session_state.logged_in = False
    st.session_state.role      = None
    st.session_state.username  = ""


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg", width=80)
    st.title("⚖️ FIR & BNS System")
    st.caption("Bharatiya Nyaya Sanhita 2023")
    st.divider()

    if not st.session_state.logged_in:
        st.subheader("🔐 Login")
        with st.form("login_form"):
            uname    = st.text_input("Username")
            pwd      = st.text_input("Password", type="password")
            role_sel = st.selectbox("Role", ["Constable", "Investigating Officer (IO)"])
            submitted = st.form_submit_button("Login")
            if submitted:
                role_key = "constable" if "Constable" in role_sel else "io"
                if login(uname, pwd, role_key):
                    st.success(f"Welcome, {uname}!")
                    st.rerun()
                else:
                    st.error("Enter valid credentials.")
    else:
        st.success(f"👤 {st.session_state.username}")
        role_label = "Constable" if st.session_state.role == "constable" else "Investigating Officer"
        st.caption(f"Role: **{role_label}**")
        st.divider()

        # Role-based navigation
        if st.session_state.role == "constable":
            page = st.radio("📂 Menu", ["Register FIR", "My FIRs", "🤖 Legal Chatbot"])
        else:
            page = st.radio("📂 Menu", ["Register FIR", "My FIRs", "Predict BNS Section", "All FIRs", "Similar Cases", "🤖 Legal Chatbot"])

        st.divider()
        if st.button("🚪 Logout"):
            logout()
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    st.title("⚖️ AI-Based FIR & Smart BNS Prediction System")
    st.info("👈 Please log in from the sidebar to continue.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("BNS Sections Covered", "312")
    with col2:
        st.metric("Top-5 Accuracy", "83.65%")
    with col3:
        st.metric("Model", "Logistic Regression")
    st.stop()


# ──────────────────────────────────────────
# PAGE: Register FIR
# ──────────────────────────────────────────
if page == "Register FIR":
    st.title("📝 Register FIR")
    st.caption("Fill in the complaint details below. All fields marked * are required.")

    with st.form("fir_form"):
        col1, col2 = st.columns(2)

        with col1:
            complainant = st.text_input("Complainant Name *")
            phone_raw   = st.text_input("Mobile Number *", placeholder="e.g. 9876543210")
            aadhar      = st.text_input("Aadhaar / ID (optional)")

        with col2:
            incident_date = st.date_input("Date of Incident *", value=datetime.today())
            incident_time = st.time_input("Time of Incident")

        # Location autocomplete
        st.subheader("📍 Incident Location")
        location_query = st.text_input("Search Location", placeholder="Type area, city, district…")
        selected_location = ""
        if location_query:
            with st.spinner("Fetching suggestions…"):
                suggestions = get_location_suggestions(location_query)
            if suggestions:
                opts = [s["display_name"] for s in suggestions]
                selected_location = st.selectbox("Select Location", opts)
            else:
                st.warning("No suggestions found. Type a more specific address.")
                selected_location = location_query

        fir_text = st.text_area(
            "Complaint / Incident Description *",
            placeholder="Describe what happened in your own words…",
            height=180
        )

        submitted = st.form_submit_button("✅ Submit FIR", use_container_width=True)

    if submitted:
        errors = []
        if not complainant.strip():
            errors.append("Complainant name is required.")
        if not is_valid_phone(phone_raw):
            errors.append("Enter a valid 10-digit Indian mobile number.")
        if not fir_text.strip():
            errors.append("Complaint description is required.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            fir_id  = f"FIR-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            fir_rec = {
                "fir_id"          : fir_id,
                "complainant"     : complainant.strip(),
                "phone"           : phone_raw.strip(),
                "incident_date"   : str(incident_date),
                "incident_time"   : str(incident_time),
                "location"        : selected_location,
                "fir_text"        : fir_text.strip(),
                "registered_by"   : st.session_state.username,
                "registered_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_sections": predict_bns(fir_text, top_k=3)
            }
            st.session_state.fir_store.append(fir_rec)

            st.success(f"✅ FIR registered successfully! ID: **{fir_id}**")
            st.balloons()

            st.subheader("🔍 Predicted BNS Sections")
            for p in fir_rec["predicted_sections"]:
                with st.expander(
                    f"BNS Section {p['section']} — {p['section_name']}  |  {p['confidence']}% confidence"
                ):
                    st.write(f"**Chapter:** {p['chapter_name']}")
                    st.write(f"**Description:** {p['description']}")


# ──────────────────────────────────────────
# PAGE: My FIRs
# ──────────────────────────────────────────
elif page == "My FIRs":
    st.title("📁 My FIRs")
    my_firs = [
        f for f in st.session_state.fir_store
        if f["registered_by"] == st.session_state.username
    ]
    if not my_firs:
        st.info("No FIRs registered yet. Go to **Register FIR** to file one.")
    else:
        st.success(f"{len(my_firs)} FIR(s) found.")
        for fir in reversed(my_firs):
            with st.expander(f"🗂 {fir['fir_id']}  |  {fir['registered_at']}"):
                st.write(f"**Complainant:** {fir['complainant']}")
                st.write(f"**Phone:** {fir['phone']}")
                st.write(f"**Date of Incident:** {fir['incident_date']} at {fir['incident_time']}")
                st.write(f"**Location:** {fir['location']}")
                st.write(f"**Complaint:** {fir['fir_text']}")
                if fir.get("predicted_sections"):
                    st.write("**Predicted BNS Sections:**")
                    for p in fir["predicted_sections"]:
                        st.write(f"  • Section {p['section']} — {p['section_name']} ({p['confidence']}%)")


# ──────────────────────────────────────────
# PAGE: Predict BNS Section  (IO only)
# ──────────────────────────────────────────
elif page == "Predict BNS Section":
    if st.session_state.role != "io":
        st.error("⛔ Access denied. This page is for Investigating Officers only.")
        st.stop()

    st.title("🔮 Predict BNS Section")
    st.caption("Enter complaint text to get AI-predicted BNS sections with confidence scores.")

    query_text = st.text_area("Complaint Text", height=150,
                              placeholder="E.g. The accused attacked me with a knife and stole my phone…")

    col1, col2 = st.columns([1, 4])
    top_k = col1.slider("Top K results", 1, 10, 5)

    if st.button("🔍 Predict", use_container_width=False):
        if not query_text.strip():
            st.warning("Please enter complaint text.")
        else:
            with st.spinner("Predicting…"):
                preds = predict_bns(query_text, top_k=top_k)

            if not preds:
                st.error("Could not generate predictions. Check the model files.")
            else:
                st.subheader("📊 Predictions")
                for i, p in enumerate(preds):
                    color = "🟢" if p["confidence"] > 30 else ("🟡" if p["confidence"] > 10 else "🔴")
                    with st.expander(
                        f"{color} #{i+1}  BNS Section {p['section']} — {p['section_name']}  |  {p['confidence']}%"
                    ):
                        st.progress(min(p["confidence"] / 100, 1.0))
                        st.write(f"**Chapter:** {p['chapter_name']}")
                        st.write(f"**Description:** {p['description']}")

                # See Precedents button
                st.divider()
                if st.button("📂 See Precedents (Similar Cases)"):
                    st.session_state["precedent_text"] = query_text
                    st.session_state["goto_similar"]   = True
                    st.rerun()


# ──────────────────────────────────────────
# PAGE: All FIRs  (IO only)
# ──────────────────────────────────────────
elif page == "All FIRs":
    if st.session_state.role != "io":
        st.error("⛔ Access denied. This page is for Investigating Officers only.")
        st.stop()

    st.title("📋 All Registered FIRs")
    all_firs = st.session_state.fir_store
    if not all_firs:
        st.info("No FIRs have been registered yet.")
    else:
        st.success(f"{len(all_firs)} total FIR(s).")
        df = pd.DataFrame([
            {
                "FIR ID"          : f["fir_id"],
                "Complainant"     : f["complainant"],
                "Date"            : f["incident_date"],
                "Location"        : f["location"][:50] + "…" if len(f.get("location","")) > 50 else f.get("location",""),
                "Registered By"   : f["registered_by"],
                "Registered At"   : f["registered_at"],
                "Top BNS Section" : (
                    f["predicted_sections"][0]["section"]
                    if f.get("predicted_sections") else "—"
                )
            }
            for f in reversed(all_firs)
        ])
        st.dataframe(df, use_container_width=True)


# ──────────────────────────────────────────
# PAGE: Similar Cases  (IO only)
# ──────────────────────────────────────────
elif page == "Similar Cases":
    if st.session_state.role != "io":
        st.error("⛔ Access denied. This page is for Investigating Officers only.")
        st.stop()

    st.title("🔗 Similar Cases — See Precedents")
    st.caption("Find past FIRs most similar to a complaint using cosine similarity.")

    # Pre-fill if routed from Predict page
    default_text = st.session_state.pop("precedent_text", "")
    query_text   = st.text_area("Complaint Text", value=default_text, height=150)
    top_n        = st.slider("Number of similar cases", 3, 20, 5)

    if st.button("🔍 Find Similar Cases", use_container_width=False):
        if not query_text or not query_text.strip():
            st.warning("Please enter complaint text.")
        else:
            with st.spinner("Computing similarity…"):
                similar_df = get_similar_firs(query_text, top_n=top_n)

            st.subheader(f"Top {top_n} Similar FIRs")
            for _, row in similar_df.iterrows():
                with st.expander(
                    f"📄 {row['fir_id']}  |  BNS §{row['bns_section']} — {row['section_name']}  |  {row['similarity']}% match"
                ):
                    st.write(row["fir_text"])


# ──────────────────────────────────────────
# PAGE: Legal Chatbot (all roles)
# ──────────────────────────────────────────
elif page == "🤖 Legal Chatbot":
    st.title("🤖 BNS Legal Assistant")
    st.caption(
        "Ask anything about BNS 2023 sections, FIR procedures, or criminal law concepts. "
        "Powered by Groq (Llama 3.3 70B) — free & fast."
    )

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=False):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    # Render existing conversation
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if prompt := st.chat_input("E.g. What is BNS Section 103? Which section covers theft?"):
        # Show user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Get and display assistant reply
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = call_groq(prompt, st.session_state.chat_history[:-1])
            st.write(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # Suggested starter questions
    if not st.session_state.chat_history:
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "What is BNS Section 103?",
            "Which BNS section applies to kidnapping?",
            "What are the steps to file an FIR?",
            "Difference between cognizable and non-cognizable offence?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": s})
                with st.spinner("Thinking…"):
                    reply = call_groq(s, [])
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AI-Based FIR & Smart Prediction System · BNS 2023 · "
    f"Model v2.0 · Top-5 Accuracy: 83.65% · {datetime.now().year}"
)