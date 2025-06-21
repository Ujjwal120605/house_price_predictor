import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="🏠 House Price Predictor", page_icon="💰")
st.title("🏠 House Price Predictor")

if not os.path.exists("house_price_model.pkl"):
    st.error("❌ Model not found. Run train_model.py first.")
    st.stop()

model = joblib.load("house_price_model.pkl")

st.header("📋 Enter House Details")

rm = st.slider("Average Rooms per Dwelling", 3.0, 9.0, 6.0)
lstat = st.slider("% of Lower Status Population", 1.0, 40.0, 12.0)
ptratio = st.slider("Pupil-Teacher Ratio", 12.0, 22.0, 18.0)

features = np.array([[rm, lstat, ptratio]])
prediction = model.predict(features)[0]

st.subheader("💸 Predicted Price")
st.success(f"${prediction * 1000:.2f}")
