**AI-Assisted FIR Registration and IPC Section Prediction System**

**Overview**

The AI-Assisted FIR Registration and IPC Section Prediction System is a decision-support system designed to assist law enforcement officers in registering First Information Reports (FIRs) digitally and predicting relevant IPC sections using Artificial Intelligence and Natural Language Processing (NLP).

The system helps reduce manual effort in analyzing FIR descriptions, improves efficiency in identifying legal sections, and assists investigation teams by providing legal information, crime severity analysis, and similar case retrieval.

This system is designed to assist officers, not replace human judgment.

**Problem Statement**

In traditional FIR systems:

FIRs are written manually.

Police officers must manually determine the relevant IPC sections.

Searching for similar past cases is difficult.

Legal references require additional time.

This project solves these problems by integrating Machine Learning and NLP into a digital FIR system.

**Key Features**

Digital FIR registration

Automatic IPC section prediction

Legal information retrieval

Crime severity classification

Similar FIR case retrieval

Investigation assistance support

System Architecture

The system consists of four main modules.

1. FIR Registration Module

Police officers enter FIR details using a digital interface.

Example input:

Complainant name

Date

Location

FIR description (natural language)

The FIR description is sent to the AI prediction engine.

2. AI Prediction Engine

The AI engine processes the FIR text using NLP techniques.

Steps performed:

Text preprocessing

TF-IDF vectorization

Machine learning classification

The trained model predicts the most relevant IPC section.

Example output:

Predicted IPC Section: 379

3. Legal Knowledge Engine

After predicting the IPC section, the system retrieves legal information from the IPC dataset.

Example:

IPC Section: 379
Offense: Theft
Punishment: Up to 3 years imprisonment

This helps officers understand the legal context of the case.

4. Investigation Assistance Module

The system also supports investigators by:

Retrieving similar FIR cases using sentence embeddings

Providing insights based on past case descriptions

Assisting investigation analysis

**AI and Machine Learning Pipeline**

The AI module performs the following steps:

FIR text preprocessing

Tokenization

Stopword removal

Stemming

Text vectorization

TF-IDF vectorizer

Classification model

Logistic Regression / Naive Bayes

Similar case retrieval

Sentence embeddings

Cosine similarity

Crime severity analysis

Based on punishment length

**Technologies Used**

Programming Language
Python

Machine Learning Libraries
scikit-learn
pandas
numpy

Natural Language Processing
nltk / spacy

Similarity Search
sentence-transformers
