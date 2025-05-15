import streamlit as st
import pandas as pd
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load data from S3 or local
@st.cache_data
def load_data():
    local_file = "combined_data.csv"
    s3_url = "https://puravida-test.s3.ap-south-1.amazonaws.com/combined_data.csv"

    if not os.path.exists(local_file):
        df = pd.read_csv(s3_url, encoding='latin1')
        df.to_csv(local_file, index=False)
    else:
        df = pd.read_csv(local_file, encoding='latin1')

    return df

# Train or load models
@st.cache_resource
def train_models():
    # Paths
    model_file = "trained_models.joblib"
    vectorizer_file = "vectorizer.joblib"
    accuracy_file = "accuracies.joblib"

    # If models already saved
    if os.path.exists(model_file) and os.path.exists(vectorizer_file) and os.path.exists(accuracy_file):
        trained_models = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)
        accuracies = joblib.load(accuracy_file)
        return trained_models, vectorizer, accuracies

    # Load and prepare data
    df = load_data()
    if df['label'].iloc[0] == 'label':
        df = df.iloc[1:]
    df['label'] = df['label'].astype(int)
    df['text'] = df['text'].astype(str)

    # Add custom offer/discount data
    custom_data = pd.DataFrame({
        'label': [2, 3, 2, 3],
        'text': [
            'Huge discount on all electronic items today! Don’t miss out!',
            'Limited time offer! 50% off on selected products.',
            'Special offer: Buy one get one free on all shoes!',
            'Massive discount this weekend on all online purchases.'
        ]
    })
    df = pd.concat([df, custom_data], ignore_index=True)
    df['text'] = df['text'].apply(clean_text)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Define models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": LinearSVC(),
        "Random Forest": RandomForestClassifier()
    }

    trained_models = {}
    accuracies = {}

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        trained_models[name] = model
        accuracies[name] = acc * 100

    # Save models, vectorizer, and accuracy
    joblib.dump(trained_models, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    joblib.dump(accuracies, accuracy_file)

    return trained_models, vectorizer, accuracies

# Prediction
def predict_spam(text_input, models, vectorizer):
    cleaned = clean_text(text_input)
    vectorized = vectorizer.transform([cleaned])

    label_map = {0: "HAM", 1: "SPAM", 2: "OFFER", 3: "DISCOUNT"}
    results = {}

    for model_name, model in models.items():
        pred = model.predict(vectorized)[0]
        result_label = label_map.get(pred, "UNKNOWN")
        results[model_name] = result_label

    return results

# Streamlit UI
st.title("📩 Email Classification with Multiple Models")
st.write("Enter a message and compare how different models classify it (SPAM, HAM, OFFER, or DISCOUNT) along with their accuracy.")

models, vectorizer, accuracies = train_models()
user_input = st.text_area("Enter your email/message:")

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        predictions = predict_spam(user_input, models, vectorizer)

        st.header("🔍 Model-wise Predictions and Accuracies")
        for model_name in models:
            pred = predictions.get(model_name, "UNKNOWN")
            acc = accuracies.get(model_name, 0.0)
            st.markdown(f"**{model_name}**")
            st.write(f"Prediction: **{pred}**")
            st.write(f"Accuracy: **{acc:.2f}%**")
            st.markdown("---")
