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

data = pd.read_csv("SPAM text message 20170820 - Data.csv")

# Text preprocessing
stemmer = PorterStemmer()
def preprocess_text(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

#data['Processed_Message'] = data['Message'].apply(preprocess_text)

# Vectorization
vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=2500, binary=True)
#`x = vectorizer.fit_transform(data['Processed_Message'][1]).toarray()

# Label Encoding
label = LabelEncoder()
y = label.fit_transform(data['Category'])

# Splitting data
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model training
#model = MultinomialNB()
#model.fit(x_train, y_train)


st.title('Spam Detection App')
st.write("This app predicts whether a given message is spam or not.")


model = joblib.load('spamdetection.joblib')

input_message = st.text_input("Enter your message:")

if st.button('Predict'):
    if input_message:
        input_message_processed = preprocess_text(input_message)
        input_vectorized = vectorizer.fit_transform([input_message_processed]).toarray()
        prediction = model.predict(input_vectorized)
        if prediction == 1:
            st.write("This is a spam message!")
        else:
            st.write("This is not a spam message.")


