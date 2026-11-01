import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = joblib.load("sentiment_model.pkl")
binary_vectorizer = joblib.load("binary_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    negation_words = {"not", "no", "never"}
    words = text.split()
    words = [
        word for word in words
        if word not in stop_words or word in negation_words
    ]

    words = [lemmatizer.lemmatize(word) for word in words]

    text = " ".join(words)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def predict_sentiment(review):
    cleaned_review = preprocess_text(review)
    review_vector = binary_vectorizer.transform([cleaned_review])
    prediction = model.predict(review_vector)[0]
    return "Positive" if prediction == 1 else "Negative"

st.set_page_config(
    page_title="Restaurant Review Sentiment Analysis",
    page_icon="🍽️"
)

st.title("🍽️ Restaurant Review Sentiment Analysis")
st.write("Enter a restaurant review and the model will predict its sentiment.")

review = st.text_area(
    "Enter your review:",
    placeholder="Example: The food was fantastic!"
)

if st.button("Predict Sentiment"):
    if review.strip():
        sentiment = predict_sentiment(review)

        if sentiment == "Positive":
            st.success("🟢 Positive Review")
        else:
            st.error("🔴 Negative Review")
    else:
        st.warning("Please enter a review first.")
