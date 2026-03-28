import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import json
import os
import warnings
from datetime import datetime, date
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI-FIR System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.main { background-color: #f0f2f6; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f36 0%, #2d3561 100%);
}
section[data-testid="stSidebar"] * { color: #ffffff !important; }
section[data-testid="stSidebar"] .stRadio label { color: #ffffff !important; }

/* Cards */
.card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    margin: 10px 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-left: 4px solid #2d3561;
}
.card-green  { border-left-color: #27ae60; }
.card-red    { border-left-color: #e74c3c; }
.card-orange { border-left-color: #e67e22; }
.card-blue   { border-left-color: #2980b9; }

/* Metric boxes */
.metric-box {
    background: linear-gradient(135deg, #2d3561, #1a1f36);
    color: white !important;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    margin: 4px;
}
.metric-box h2 { color: #f0c040 !important; font-size: 2rem; margin: 0; }
.metric-box p  { color: #ccc !important; margin: 0; font-size: 0.85rem; }

/* Severity badges */
.sev-low      { background:#27ae60; color:white; padding:4px 14px; border-radius:20px; font-weight:600; }
.sev-medium   { background:#f39c12; color:white; padding:4px 14px; border-radius:20px; font-weight:600; }
.sev-high     { background:#e67e22; color:white; padding:4px 14px; border-radius:20px; font-weight:600; }
.sev-critical { background:#e74c3c; color:white; padding:4px 14px; border-radius:20px; font-weight:600; }

/* IPC result box */
.ipc-result {
    background: linear-gradient(135deg, #1a1f36, #2d3561);
    color: white;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}
.ipc-result h1 { color: #f0c040; font-size: 3rem; margin: 0; }
.ipc-result h3 { color: #fff; margin: 4px 0; }
.ipc-result p  { color: #aab; margin: 0; }

/* FIR card in history */
.fir-card {
    background: white;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.1);
    border-left: 4px solid #2d3561;
}

/* Button override */
.stButton > button {
    background: linear-gradient(135deg, #2d3561, #1a1f36);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    width: 100%;
}
.stButton > button:hover { background: #e74c3c; color: white; }

/* Page header */
.page-header {
    background: linear-gradient(135deg, #1a1f36 0%, #2d3561 100%);
    color: white;
    padding: 24px 28px;
    border-radius: 12px;
    margin-bottom: 24px;
}
.page-header h2 { color: #f0c040; margin: 0 0 4px 0; }
.page-header p  { color: #aab; margin: 0; font-size: 0.9rem; }

/* Similar FIR card */
.similar-card {
    background: #f8f9ff;
    border: 1px solid #d0d6f0;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
}
.similar-card .score { color: #2d3561; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA / MODEL PATHS  (relative to this script)
# ─────────────────────────────────────────────
BASE = os.path.dirname(__file__)
AI   = os.path.join(BASE, "ai_module")

MODEL_PATH      = os.path.join(AI, "models", "ipc_model.pkl")
VECTORIZER_PATH = os.path.join(AI, "models", "tfidf_vectorizer.pkl")
FIR_DATA_PATH   = os.path.join(AI, "datasets", "fir_dataset_with_id.csv")
IPC_DATA_PATH   = os.path.join(AI, "datasets", "ipc_sections.csv")
FIR_DB_PATH     = os.path.join(BASE, "registered_firs.json")

# ─────────────────────────────────────────────
# STOPWORDS (inline — no NLTK download needed)
# ─────────────────────────────────────────────
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll","m",
    "o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
    "weren","won","wouldn"
}

# ─────────────────────────────────────────────
# LOAD MODELS & DATA
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

@st.cache_data
def load_data():
    fir_df = pd.read_csv(FIR_DATA_PATH)
    ipc_df = pd.read_csv(IPC_DATA_PATH)
    return fir_df, ipc_df

# ─────────────────────────────────────────────
# HELPERS  (defined before cache functions that call them)
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    text  = text.lower()
    text  = re.sub(r"[^a-zA-Z]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)

@st.cache_data
def build_fir_tfidf_matrix(_vectorizer, fir_texts):
    """Pre-compute TF-IDF matrix for all FIRs for similarity search."""
    processed = [preprocess_text(t) for t in fir_texts]
    return _vectorizer.transform(processed)

model, vectorizer = load_models()
fir_df, ipc_df   = load_data()
FIR_MATRIX       = build_fir_tfidf_matrix(vectorizer, fir_df["fir_text"].tolist())

def predict_ipc(description: str) -> int:
    processed = preprocess_text(description)
    vector    = vectorizer.transform([processed])
    return int(model.predict(vector)[0])

def get_ipc_details(section: int):
    key = f"IPC_{section}"
    row = ipc_df[ipc_df["Section"] == key]
    if row.empty:
        return None, None, None
    offense    = row["Offense"].values[0]
    punishment = row["Punishment"].values[0]
    description = row["Description"].values[0] if "Description" in row.columns else ""
    return offense, punishment, description

def extract_years(punishment_text: str) -> int:
    pt = str(punishment_text).lower()
    if "life imprisonment" in pt or "life" in pt:
        return 25
    nums = re.findall(r"\d+", pt)
    return max(map(int, nums)) if nums else 0

def get_severity(punishment_text: str):
    years = extract_years(punishment_text)
    if years <= 2:
        return "Low", "🟢", "sev-low"
    elif years <= 5:
        return "Medium", "🟡", "sev-medium"
    elif years <= 10:
        return "High", "🟠", "sev-high"
    else:
        return "Critical", "🔴", "sev-critical"

def find_similar_firs(description: str, top_k: int = 5):
    processed = preprocess_text(description)
    q_vec     = vectorizer.transform([processed])
    scores    = cosine_similarity(q_vec, FIR_MATRIX)[0]
    top_idx   = scores.argsort()[::-1][:top_k]
    results   = fir_df.iloc[top_idx].copy()
    results["similarity"] = scores[top_idx]
    return results[results["similarity"] > 0.05]

def load_registered_firs():
    if os.path.exists(FIR_DB_PATH):
        with open(FIR_DB_PATH) as f:
            return json.load(f)
    return []

def save_fir(record: dict):
    firs = load_registered_firs()
    firs.append(record)
    with open(FIR_DB_PATH, "w") as f:
        json.dump(firs, f, indent=2)

def generate_fir_number():
    firs = load_registered_firs()
    return f"FIR-{datetime.now().strftime('%Y%m%d')}-{len(firs)+1:04d}"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px 0;'>
        <div style='font-size:3rem;'>⚖️</div>
        <h2 style='margin:0; color:#f0c040;'>AI-FIR System</h2>
        <p style='color:#aab; font-size:0.8rem; margin:4px 0 0 0;'>
            Smart Prediction & Investigation
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠  Dashboard", "📝  Register FIR", "🔍  Predict IPC", "📂  FIR Records", "🔎  Similar Cases"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    registered_firs = load_registered_firs()
    st.markdown(f"""
    <div style='text-align:center;'>
        <p style='color:#aab; font-size:0.8rem; margin:0;'>Total FIRs Registered</p>
        <h2 style='color:#f0c040; margin:4px 0;'>{len(registered_firs)}</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='color:#667; font-size:0.75rem; text-align:center;'>Powered by ML + NLP<br>For Law Enforcement Use</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────
if page == "🏠  Dashboard":
    st.markdown("""
    <div class='page-header'>
        <h2>⚖️ AI-Assisted FIR Registration System</h2>
        <p>Intelligent crime reporting with automatic IPC section prediction and investigation support</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    registered_firs = load_registered_firs()
    sections_used = set(f.get("predicted_ipc") for f in registered_firs if f.get("predicted_ipc"))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-box'><h2>{len(registered_firs)}</h2><p>FIRs Registered</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-box'><h2>{len(sections_used)}</h2><p>IPC Sections Used</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-box'><h2>{len(fir_df)}</h2><p>Training Cases</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-box'><h2>{len(ipc_df)}</h2><p>IPC Sections in DB</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='card card-blue'>
            <h3>📝 Digital FIR Registration</h3>
            <p>Register FIRs digitally with complainant details, location, date and full description. Each FIR gets a unique ID.</p>
        </div>
        <div class='card card-green'>
            <h3>🤖 AI IPC Prediction</h3>
            <p>Automatically predicts the most relevant IPC section using TF-IDF vectorization and Logistic Regression trained on 800+ FIR cases.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card card-orange'>
            <h3>⚖️ Legal Knowledge Engine</h3>
            <p>Instantly retrieves offense description, punishment details and crime severity classification for any predicted IPC section.</p>
        </div>
        <div class='card card-red'>
            <h3>🔎 Similar Case Retrieval</h3>
            <p>Finds the most similar past FIR cases using cosine similarity to assist investigators with precedent-based insights.</p>
        </div>
        """, unsafe_allow_html=True)

    # Recent FIRs
    if registered_firs:
        st.markdown("### 📋 Recent FIR Activity")
        recent = registered_firs[-5:][::-1]
        for fir in recent:
            sev_label, sev_icon, _ = get_severity(fir.get("punishment", ""))
            st.markdown(f"""
            <div class='fir-card'>
                <b>{fir.get('fir_number','—')}</b> &nbsp;|&nbsp;
                <b>IPC {fir.get('predicted_ipc','?')}</b> &nbsp;|&nbsp;
                {fir.get('offense','—')} &nbsp;|&nbsp;
                {sev_icon} {sev_label} &nbsp;|&nbsp;
                📍 {fir.get('location','—')} &nbsp;|&nbsp;
                🗓 {fir.get('date','—')}
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: REGISTER FIR
# ─────────────────────────────────────────────
elif page == "📝  Register FIR":
    st.markdown("""
    <div class='page-header'>
        <h2>📝 Register New FIR</h2>
        <p>Enter complainant details and FIR description — IPC section will be predicted automatically</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("fir_form", clear_on_submit=True):
        st.markdown("#### 👤 Complainant Information")
        col1, col2 = st.columns(2)
        with col1:
            complainant_name  = st.text_input("Complainant Name *", placeholder="e.g. Rajesh Kumar")
            complainant_phone = st.text_input("Phone Number", placeholder="e.g. 9876543210")
        with col2:
            complainant_age     = st.number_input("Age", min_value=1, max_value=120, value=30)
            complainant_address = st.text_input("Address", placeholder="Full address")

        st.markdown("#### 📍 Incident Details")
        col3, col4 = st.columns(2)
        with col3:
            incident_date     = st.date_input("Date of Incident *", value=date.today())
            incident_location = st.text_input("Location / Place of Incident *", placeholder="e.g. MG Road, Pune")
        with col4:
            incident_time     = st.time_input("Time of Incident")
            officer_name      = st.text_input("Reporting Officer", placeholder="Officer name / badge")

        st.markdown("#### 📄 FIR Description")
        fir_description = st.text_area(
            "Describe the incident in detail *",
            height=150,
            placeholder="e.g. My motorcycle was stolen from the parking area outside my office at MG Road. I had parked the vehicle at 9 AM and when I returned at 6 PM, it was missing. The vehicle is a Honda Activa, black colour, registration MH-12-AB-1234.",
        )

        submitted = st.form_submit_button("🚀 Register FIR & Predict IPC Section")

    if submitted:
        if not complainant_name.strip() or not fir_description.strip() or not incident_location.strip():
            st.error("⚠️ Please fill in all required fields (marked with *).")
        elif len(fir_description.strip()) < 20:
            st.error("⚠️ FIR description is too short. Please provide more details.")
        else:
            with st.spinner("🤖 AI is analyzing and predicting IPC section..."):
                predicted_section = predict_ipc(fir_description)
                offense, punishment, desc = get_ipc_details(predicted_section)
                severity, sev_icon, sev_class = get_severity(punishment or "")
                fir_number = generate_fir_number()

                record = {
                    "fir_number":     fir_number,
                    "complainant":    complainant_name,
                    "phone":          complainant_phone,
                    "age":            complainant_age,
                    "address":        complainant_address,
                    "date":           str(incident_date),
                    "time":           str(incident_time),
                    "location":       incident_location,
                    "officer":        officer_name,
                    "description":    fir_description,
                    "predicted_ipc":  predicted_section,
                    "offense":        offense or "Unknown",
                    "punishment":     punishment or "Unknown",
                    "severity":       severity,
                    "registered_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_fir(record)

            st.success(f"✅ FIR Registered Successfully! Number: **{fir_number}**")

            # IPC result
            st.markdown(f"""
            <div class='ipc-result'>
                <p>Predicted IPC Section</p>
                <h1>§ {predicted_section}</h1>
                <h3>{offense or 'Unknown Offense'}</h3>
                <p>Punishment: {punishment or 'Refer legal database'}</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class='card card-blue'><h4>📋 FIR Number</h4><h3>{fir_number}</h3></div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class='card card-orange'><h4>⚖️ Offense</h4><h3>{offense or '—'}</h3></div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class='card {"card-red" if severity in ["High","Critical"] else "card-green"}'><h4>🚨 Severity</h4><h3>{sev_icon} {severity}</h3></div>""", unsafe_allow_html=True)

            # Similar cases preview
            st.markdown("#### 🔎 Similar Past Cases")
            similar = find_similar_firs(fir_description, top_k=3)
            if not similar.empty:
                for _, row in similar.iterrows():
                    score_pct = int(row["similarity"] * 100)
                    st.markdown(f"""
                    <div class='similar-card'>
                        <span class='score'>🔗 {score_pct}% match</span> &nbsp;|&nbsp;
                        <b>IPC {row['ipc_section']}</b><br>
                        <small>{str(row['fir_text'])[:200]}...</small>
                    </div>
                    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: PREDICT IPC (standalone tool)
# ─────────────────────────────────────────────
elif page == "🔍  Predict IPC":
    st.markdown("""
    <div class='page-header'>
        <h2>🔍 IPC Section Predictor</h2>
        <p>Enter any crime description and get instant IPC section prediction with legal details</p>
    </div>
    """, unsafe_allow_html=True)

    description = st.text_area(
        "Enter FIR / Crime Description",
        height=140,
        placeholder="Describe the crime or incident here...",
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        predict_clicked = st.button("🔍 Predict IPC")

    if predict_clicked:
        if not description.strip() or len(description.strip()) < 10:
            st.error("Please enter a meaningful description.")
        else:
            with st.spinner("Analyzing..."):
                section = predict_ipc(description)
                offense, punishment, ipc_desc = get_ipc_details(section)
                severity, sev_icon, sev_class = get_severity(punishment or "")

            st.markdown(f"""
            <div class='ipc-result'>
                <p style='color:#aab;'>Predicted IPC Section</p>
                <h1>§ {section}</h1>
                <h3>{offense or 'Unknown'}</h3>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class='card'><h4>⚖️ Offense</h4><p>{offense or '—'}</p></div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class='card'><h4>🏛️ Punishment</h4><p>{punishment or '—'}</p></div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class='card'><h4>🚨 Severity</h4><p><span class='{sev_class}'>{sev_icon} {severity}</span></p></div>""", unsafe_allow_html=True)

            if ipc_desc:
                with st.expander("📖 Full IPC Section Description"):
                    st.write(ipc_desc)

            # Confidence — top 3 predictions
            st.markdown("#### 📊 Top IPC Section Predictions")
            processed = preprocess_text(description)
            vec       = vectorizer.transform([processed])
            proba     = model.predict_proba(vec)[0]
            top3_idx  = proba.argsort()[::-1][:3]
            classes   = model.classes_

            for i, idx in enumerate(top3_idx):
                sec  = classes[idx]
                prob = proba[idx]
                off, pun, _ = get_ipc_details(int(sec))
                label = f"IPC {sec} — {off or 'Unknown'}"
                st.progress(float(prob), text=f"{'🥇' if i==0 else '🥈' if i==1 else '🥉'} {label}  ({prob*100:.1f}%)")

# ─────────────────────────────────────────────
# PAGE: FIR RECORDS
# ─────────────────────────────────────────────
elif page == "📂  FIR Records":
    st.markdown("""
    <div class='page-header'>
        <h2>📂 Registered FIR Records</h2>
        <p>View and search all FIRs registered through the system</p>
    </div>
    """, unsafe_allow_html=True)

    firs = load_registered_firs()

    if not firs:
        st.info("No FIRs registered yet. Go to **Register FIR** to add one.")
    else:
        # Search / filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search = st.text_input("🔍 Search by FIR number, complainant, location, or IPC section", placeholder="e.g. Pune or IPC 379")
        with col2:
            sev_filter = st.selectbox("Filter by Severity", ["All", "Critical", "High", "Medium", "Low"])

        filtered = firs
        if search:
            q = search.lower()
            filtered = [f for f in filtered if
                q in f.get("fir_number","").lower() or
                q in f.get("complainant","").lower() or
                q in f.get("location","").lower() or
                q in str(f.get("predicted_ipc","")).lower()
            ]
        if sev_filter != "All":
            filtered = [f for f in filtered if f.get("severity","") == sev_filter]

        st.markdown(f"**{len(filtered)} record(s) found**")

        for fir in reversed(filtered):
            sev_label, sev_icon, sev_class = get_severity(fir.get("punishment",""))
            with st.expander(f"📄 {fir.get('fir_number','—')}  |  {fir.get('complainant','—')}  |  IPC {fir.get('predicted_ipc','?')}  |  {sev_icon} {sev_label}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**FIR Number:** {fir.get('fir_number','—')}")
                    st.markdown(f"**Complainant:** {fir.get('complainant','—')}")
                    st.markdown(f"**Phone:** {fir.get('phone','—')}")
                    st.markdown(f"**Age:** {fir.get('age','—')}")
                with col2:
                    st.markdown(f"**Date:** {fir.get('date','—')}")
                    st.markdown(f"**Time:** {fir.get('time','—')}")
                    st.markdown(f"**Location:** {fir.get('location','—')}")
                    st.markdown(f"**Officer:** {fir.get('officer','—')}")
                with col3:
                    st.markdown(f"**IPC Section:** {fir.get('predicted_ipc','—')}")
                    st.markdown(f"**Offense:** {fir.get('offense','—')}")
                    st.markdown(f"**Punishment:** {fir.get('punishment','—')}")
                    st.markdown(f"**Severity:** {sev_icon} {sev_label}")
                st.markdown("**Description:**")
                st.info(fir.get("description","—"))

        # Download
        if filtered:
            df_export = pd.DataFrame(filtered)
            csv = df_export.to_csv(index=False)
            st.download_button(
                "⬇️ Download Records as CSV",
                csv,
                file_name=f"fir_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

# ─────────────────────────────────────────────
# PAGE: SIMILAR CASES
# ─────────────────────────────────────────────
elif page == "🔎  Similar Cases":
    st.markdown("""
    <div class='page-header'>
        <h2>🔎 Similar Case Retrieval</h2>
        <p>Find similar past FIR cases to assist investigation using AI-powered text similarity</p>
    </div>
    """, unsafe_allow_html=True)

    description = st.text_area(
        "Enter crime description to find similar cases",
        height=130,
        placeholder="e.g. Someone broke into my shop at night and stole cash from the counter...",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        top_k = st.slider("Number of similar cases", 3, 15, 5)
    with col2:
        search_clicked = st.button("🔎 Find Similar Cases")

    if search_clicked:
        if not description.strip() or len(description.strip()) < 10:
            st.error("Please enter a meaningful description.")
        else:
            with st.spinner("Searching through case database..."):
                results = find_similar_firs(description, top_k=top_k)

            if results.empty:
                st.warning("No sufficiently similar cases found.")
            else:
                st.markdown(f"### Found {len(results)} Similar Cases")

                for rank, (_, row) in enumerate(results.iterrows(), 1):
                    score_pct = int(row["similarity"] * 100)
                    section   = int(row["ipc_section"])
                    offense, punishment, _ = get_ipc_details(section)
                    sev_label, sev_icon, _ = get_severity(punishment or "")

                    # Color bar based on score
                    bar_color = "#27ae60" if score_pct > 60 else "#f39c12" if score_pct > 30 else "#e74c3c"

                    st.markdown(f"""
                    <div class='fir-card'>
                        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
                            <b>#{rank} — Case ID: {row['fir_id']}</b>
                            <span style='background:{bar_color}; color:white; padding:2px 12px; border-radius:20px; font-size:0.85rem;'>
                                {score_pct}% match
                            </span>
                        </div>
                        <p style='margin:0 0 8px 0; color:#333;'>{str(row['fir_text'])}</p>
                        <small>
                            <b>IPC {section}</b> — {offense or 'Unknown'} &nbsp;|&nbsp;
                            🏛️ {punishment or '—'} &nbsp;|&nbsp;
                            {sev_icon} {sev_label}
                        </small>
                    </div>
                    """, unsafe_allow_html=True)

                # Insight summary
                st.markdown("---")
                st.markdown("### 💡 Investigation Insights")
                most_common_ipc = results["ipc_section"].mode()[0]
                off_c, pun_c, _ = get_ipc_details(int(most_common_ipc))
                sev_c, sev_icon_c, _ = get_severity(pun_c or "")
                avg_sim = results["similarity"].mean()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""<div class='card card-blue'>
                        <h4>Most Common IPC Section</h4>
                        <h3>§ {most_common_ipc}</h3>
                        <p>{off_c or '—'}</p>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""<div class='card card-orange'>
                        <h4>Avg. Case Similarity</h4>
                        <h3>{avg_sim*100:.1f}%</h3>
                        <p>across top {len(results)} cases</p>
                    </div>""", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""<div class='card {"card-red" if sev_c in ["High","Critical"] else "card-green"}'>
                        <h4>Likely Crime Severity</h4>
                        <h3>{sev_icon_c} {sev_c}</h3>
                        <p>based on similar cases</p>
                    </div>""", unsafe_allow_html=True)
