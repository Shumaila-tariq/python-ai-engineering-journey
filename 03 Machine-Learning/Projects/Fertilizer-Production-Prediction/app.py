import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Fertilizer Production Predictor",
    page_icon="🌱",
    layout="centered"
)

# Load model
model = joblib.load("model.pkl")

st.title("🌱 Fertilizer Production Prediction")

st.markdown(
    "Enter a **year** below to predict fertilizer production using a Linear Regression model."
)

year = st.number_input(
    "Year",
    min_value=2000,
    max_value=2050,
    value=2025
)

if st.button("Predict"):

    prediction = model.predict(np.array([[year]]))

    st.success(
        f"🌾 Predicted Fertilizer Production: **{prediction[0]:.2f}**"
    )

st.divider()

st.caption("Developed using Streamlit & Scikit-learn")