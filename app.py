import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # added to fix Streamlit Cloud error

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stop_words and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    #1. preprocess
    transform_sms = transform_text(input_sms)
    #2. vectorize
    vector_input = tfidf.transform([transform_sms])
    #3. predict
    result= model.predict(vector_input)[0]
    #4. display
    if result == 1:
        st.header("Spam")
        st.write("⚠️ This message looks like spam. Be cautious before clicking any links or sharing information.")
    else:
        st.header("Not Spam")
        st.write("✅ This message seems safe. No signs of spam detected.")

