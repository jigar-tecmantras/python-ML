import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load and prepare model (only once)
@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['text'] = df['text'].apply(clean_text)
    
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer

model, vectorizer = train_model()

# Define a function to make predictions
def predict_spam(text_input):
    cleaned = clean_text(text_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "SPAM" if prediction == 1 else "Not Spam"

# Streamlit interface
st.title("📩 Spam Detector")
st.write("Enter a message below to check if it's SPAM or HAM.")

user_input = st.text_area("Enter your message:")

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        result = predict_spam(user_input)
        st.success(f"This message is classified as: **{result}**")
