import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import nltk


stemmer = PorterStemmer()
def preprocess_text(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text



vectorizer = joblib.load('countvectorizer.joblib')
st.title('Spam Detection App')
st.write("This app predicts whether a given message is spam or not.")


model = joblib.load('spamdetection.joblib')

input_message = st.text_input("Enter your message:")

if st.button('Predict'):
    if input_message:
        input_message_processed = preprocess_text(input_message)
        input_vectorized = vectorizer.transform([input_message_processed]).toarray()

        prediction = model.predict(input_vectorized)
        if prediction >= 0.5:
            st.write("This is a spam message!")
        else:
            st.write("This is not a spam message.")


