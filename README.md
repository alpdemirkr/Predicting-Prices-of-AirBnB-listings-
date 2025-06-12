
# 🏡 Airbnb Price Prediction – London Listings

**Author:** Alpdemir Kör  
**Objective:** Predict the nightly price of Airbnb listings in London using structured features and natural language processing (NLP) on listing descriptions and titles.

---

## 📌 Project Overview

This project explores and models Airbnb listing prices in London. The objective is to build both benchmark and deep learning models that capture patterns in structured data (e.g., number of bedrooms, location, etc.) as well as unstructured data (listing title and description).

Key goals:
- Clean and prepare listing data for machine learning.
- Engineer meaningful features from both structured and text-based data.
- Apply EDA to reveal drivers of price variation.
- Build a Long Short-Term Memory (LSTM) neural network leveraging Word2Vec embeddings for NLP features.
- Benchmark against a Random Forest regressor.

---

## 📊 Dataset

The dataset includes structured and unstructured information from Airbnb listings, such as:
- Listing metadata (e.g., number of bedrooms, bathrooms, accommodates).
- Host-related attributes (e.g., superhost status, verification).
- Location-level attributes (e.g., neighborhood, property type).
- Free-text listing fields (`name`, `description`).

> **Note:** The original dataset files (`trn.csv`, `test.csv`, etc.) are not included in this repository. A summary is provided in `data/README.md`.

---

## 🔍 Methodology

### 1. Data Cleaning & EDA
- Removed irrelevant or low-variance features (e.g., zip code, photo links).
- Replaced missing values using appropriate strategies (median, mode).
- Detected and treated outliers using Tukey’s rule.
- Visualized price distributions by room type, property type, and neighborhood.

### 2. Text Preprocessing
- Created a custom NLP pipeline using `nltk`.
- Tokenized, stemmed, and removed stopwords from both `name` and `description` fields.
- Generated word clouds to explore text data.
- Trained a Word2Vec model to capture semantic relationships.

### 3. Modeling Approaches
- **LSTM model** trained on tokenized and padded sequences with pretrained embeddings.
- **Random Forest** trained on structured variables, used as a benchmark.

### 4. Evaluation
- Mean Squared Error (MSE) used for model evaluation.
- Model predictions visualized and compared to actual values.

---

## ⚙️ Tools & Libraries

- **Pandas / NumPy / Matplotlib / Seaborn** – Data processing and visualization
- **NLTK / Gensim** – Natural Language Processing and Word2Vec modeling
- **Scikit-learn** – Regression modeling and preprocessing
- **Keras / TensorFlow** – Neural network architecture (LSTM)

---

