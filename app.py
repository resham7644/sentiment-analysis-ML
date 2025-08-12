# app.py
import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review and see if it's Positive or Negative.")

# Input box
review = st.text_area("Type your review here...")

if st.button("Analyze Sentiment"):
    if review.strip():
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]
        st.write("**Prediction:**", prediction)
    else:
        st.warning("Please enter a review text.")
