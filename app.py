
import streamlit as st
import pickle
import pandas as pd

# Load trained model and selected features
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("selected_features.pkl", "rb") as file:
    selected_features = pickle.load(file)

st.title("Phishing Website Detection System")

st.write("""
This application predicts whether a website is likely to be phishing or legitimate
based on selected URL and webpage-related features.
""")

st.subheader("Enter Website Feature Values")

input_data = {}

for feature in selected_features:
    input_data[feature] = st.number_input(
        label=feature,
        value=0.0,
        step=1.0
    )

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    if prediction == 0:
        st.error("Prediction: Phishing Website")
    else:
        st.success("Prediction: Legitimate Website")
