import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Enhanced clean text function
def clean_text(text):
    text = text.lower() 
    text = re.sub(r'\d+', '', text) 
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\s+', ' ', text)  
    return text.strip()

# Train models
@st.cache_resource
def train_models():
    # Read CSV directly from S3 URL
    df = pd.read_csv("https://puravida-test.s3.ap-south-1.amazonaws.com/combined_data.csv", encoding='latin1')

    # If 'label' is a string (i.e., column name), drop the first row assuming it's a bad header
    if df['label'].iloc[0] == 'label':
        df = df.iloc[1:]

    # Ensure correct dtypes
    df['label'] = df['label'].astype(int)
    df['text'] = df['text'].astype(str)

    # Adding custom "offer" and "discount" data with numeric labels 2 and 3
    custom_data = pd.DataFrame({
        'label': [2, 3, 2, 3],
        'text': [
            'Huge discount on all electronic items today! Don’t miss out!',
            'Limited time offer! 50% off on selected products.',
            'Special offer: Buy one get one free on all shoes!',
            'Massive discount this weekend on all online purchases.'
        ]
    })

    # Combine datasets
    df = pd.concat([df, custom_data], ignore_index=True)

    # Clean text
    df['text'] = df['text'].apply(clean_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Models
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
