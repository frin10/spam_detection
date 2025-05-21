import streamlit as st
import pickle
from utils import preprocess_text

# Load the trained model and vectorizer
with open("spam_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit app layout
st.set_page_config(page_title="Spam Detector", layout="centered")
st.title("📧 Spam or Ham Email Classifier")

st.markdown("Enter the email message below to check if it's spam.")

# Input from user
user_input = st.text_area("Email Text", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess and transform input
        cleaned = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(vectorized_input)[0]
        label = "🚫 Spam" if prediction == 1 else "✅ Ham (Not Spam)"

        # Output
        st.subheader("Prediction Result:")
        st.success(label)
