import streamlit as st
import nltk
nltk.download('all')
from nltk import PorterStemmer
import pickle
from nltk.corpus import stopwords


# Note:
# To launch the web app, run the following in terminal
#  streamlit run web_app.py



st.set_page_config(page_title="SMS Spam Detection", page_icon="📬")

def transform_text(text):
    words = nltk.word_tokenize(text.lower())
    
    ps = PorterStemmer()
         
    return " ".join([ps.stem(word) for word in words if word.isalnum() and \
                    word not in stopwords.words('english')])

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')
user_input = st.text_area('Enter message here:')



if st.button('Predict'):
    transformed_text = transform_text(user_input)
    
    vectorized_text = tfidf.transform([transformed_text])
    
    result = model.predict(vectorized_text)[0]
    
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')