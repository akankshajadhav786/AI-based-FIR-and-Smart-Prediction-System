# AI-Based FIR and Smart Prediction System

A Streamlit web application for digital FIR registration with automatic IPC section prediction.

## Project Structure

```
fir_app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
└── ai_module/
    ├── datasets/
    │   ├── fir_dataset_with_id.csv # 800 FIR training cases
    │   └── ipc_sections.csv        # IPC legal database
    └── models/
        ├── ipc_model.pkl           # Trained Logistic Regression model
        ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
        └── fir_embeddings (1).pkl  # Sentence embeddings (reference)
```

## How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Launch the App
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

## Features

| Module | Description |
|--------|-------------|
| 🏠 Dashboard | Overview, stats, recent FIR activity |
| 📝 Register FIR | Digital FIR form with auto IPC prediction |
| 🔍 Predict IPC | Standalone IPC predictor with confidence scores |
| 📂 FIR Records | Search, view, and export all registered FIRs |
| 🔎 Similar Cases | Find similar past cases for investigation |

## System Architecture

1. **FIR Registration Module** — Digital form capturing complainant, location, date, description
2. **AI Prediction Engine** — TF-IDF vectorization + Logistic Regression (trained on 800 FIRs)
3. **Legal Knowledge Engine** — IPC section lookup (offense, punishment, severity)
4. **Investigation Assistance** — Cosine similarity search over the FIR case database

## Notes

- No NLTK download required — stopwords are bundled inline
- FIRs are saved locally in `registered_firs.json`
- `sentence-transformers` not required — similarity uses TF-IDF cosine similarity
