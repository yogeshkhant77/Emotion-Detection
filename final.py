import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict_sentiment(review):
    try:
        # Ensure input review is in string format
        review = str(review)
        # Transform the review using TF-IDF vectorizer
        review_tfidf = tfidf_vectorizer.transform([review])
        # Predict sentiment using the model
        prediction = model.predict(review_tfidf)[0]
        return prediction
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit UI
st.title('Sentiment Analysis')
st.image('bg.jpg')
st.write("\n\n")

review_input = st.text_area('Enter your review:')
if st.button('Predict'):
    if review_input:
        sentiment = predict_sentiment(review_input)
        if sentiment == 1:
            st.write('Negative sentiment')
        elif sentiment == 2:
            st.write('Positive sentiment')
    else:
        st.write('Please enter a review to predict sentiment.')

