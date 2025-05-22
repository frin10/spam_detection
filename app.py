import streamlit as st
import pickle
from utils import preprocess_text

# Load model and vectorizer
with open("spam_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📧 Spam or Ham Email Classifier")

user_input = st.text_area("Enter your email text here:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        label = "🚫 Spam" if prediction == 1 else "✅ Ham (Not Spam)"
        st.success(label)

        st.write(f"**Spam Probability:** {proba[1]*100:.2f}%")
        st.write(f"**Ham Probability:** {proba[0]*100:.2f}%")
