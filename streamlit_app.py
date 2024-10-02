# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# ps = PorterStemmer()

# nltk.download('punkt_tab')
# nltk.download('stopwords')
# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

# st.title("AI vs Human Detector")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Student")
#     else:
#         st.header("AI")
# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer
#
# ps = PorterStemmer()
#
#
# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)
#
#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#
#     text = y[:]
#     y.clear()
#
#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)
#
#     text = y[:]
#     y.clear()
#
#     for i in text:
#         y.append(ps.stem(i))
#
#     return " ".join(y)
#
# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))
#
# st.title("AI vs Human Detector")
# count=0
# input_sms = st.text_area("Enter the message")
# for i in input_sms:
#     if (i!=0):
#         count=count+1
# if st.button('Predict'):
#
#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Student")
#     else:
#         st.header("AI")


import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PIL import Image

# Initialize PorterStemmer
ps = PorterStemmer()


# Function to transform the text (preprocess)
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)


# Load the saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Title of the app
st.markdown("""
    <h1 style="
        text-align: center; 
        background: linear-gradient(to right, #ff4d4d, #ff4d4d,#ffa500, #ffffff); 
        -webkit-background-clip: text;
        color: transparent;
    ">
    AI vs Human Text Detector
    </h1>
    """, unsafe_allow_html=True)

# Input box
input_sms = st.text_area("Enter the message", placeholder="Type here... (at least 10 words)", height=150)

# Word count display
if input_sms:
    word_count = len(input_sms.split())
    st.markdown(f"<h5 style='text-align: right;'>Word count: {word_count}</h5>", unsafe_allow_html=True)

# Prediction button
if st.button('Predict'):
    if len(input_sms.split()) < 10:
        st.error("Please enter a message with at least 10 words.")
    else:
        # 1. Preprocess the text
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize the text
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict and get confidence scores
        result = model.predict(vector_input)[0]
        confidence_scores = model.predict_proba(vector_input)[0]
        confidence_student = confidence_scores[1] * 100
        confidence_ai = confidence_scores[0] * 100

        # 4. Display results in a visually appealing way
        if result == 1:
            st.markdown(
                f"<div style='padding: 20px; background-color: #d4edda; border-radius: 10px; border: 2px solid #155724;'>"
                f"<h3 style='text-align: center; color: #155724;'>Prediction: Student</h3>"
                f"</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='padding: 20px; background-color: #f8d7da; border-radius: 10px; border: 2px solid #721c24;'>"
                f"<h3 style='text-align: center; color: #721c24;'>Prediction: AI</h3>"
                f"</div>", unsafe_allow_html=True)

        st.markdown("<h4 style='text-align: center;'>Confidence Levels</h4>", unsafe_allow_html=True)

        # Confidence bars
        st.progress(confidence_student / 100)
        st.markdown(f"<p style='text-align: center;'><b>Student: {confidence_student:.2f}%</b></p>",
                    unsafe_allow_html=True)

        st.progress(confidence_ai / 100)
        st.markdown(f"<p style='text-align: center;'><b>AI: {confidence_ai:.2f}%</b></p>", unsafe_allow_html=True)

# Custom CSS for background and fonts
