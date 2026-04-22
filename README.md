# AI-Assisted FIR Registration and BNS Section Prediction System

## Overview

An intelligent decision-support platform that modernizes FIR registration using AI, NLP, and Machine Learning. The system helps law enforcement personnel register FIRs, predict relevant Bharatiya Nyaya Sanhita (BNS) sections, and retrieve similar historical cases.

## Features

* Smart FIR Registration
* BNS Section Prediction (Top-K predictions with confidence)
* Similar Case Retrieval using Cosine Similarity
* Role-Based Access Control (Constable / Investigation Officer)
* Location Autocomplete with OpenStreetMap Nominatim API
* Phone Number Validation
* FIR Records Management

## Tech Stack

* Frontend: Streamlit
* Backend: Python
* ML: Scikit-learn (Logistic Regression, TF-IDF)
* Data: Pandas, NumPy, CSV
* NLP: Regex / Text Preprocessing
* APIs: OpenStreetMap Nominatim, Groq API
* Version Control: Git, GitHub

## Model Performance

* Top-1 Accuracy: 49.68%
* Top-3 Accuracy: 76.28%
* Top-5 Accuracy: 83.65%
* Classes: 312 Active BNS Sections

## Installation

1. Clone the repository
2. Create a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

## Usage

1. Login as Constable or Investigation Officer.
2. Register FIR using complaint details.
3. Predict BNS sections.
4. Explore similar past FIRs.
5. Manage records.

## Future Scope

* Multilingual support
* Transformer-based models
* CCTNS integration
* Mobile app
* Continuous learning from officer feedback

## License

For academic and educational purposes.
