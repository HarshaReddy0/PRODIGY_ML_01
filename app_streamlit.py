import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("house_price_model.pkl", "rb"))

# App title
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
st.title("🏠 House Price Prediction App")
st.write("Predict house prices using Linear Regression")

st.sidebar.header("Enter House Details")

# User Inputs
sqft = st.sidebar.number_input(
    "Square Footage (sq ft)", min_value=300, max_value=5000, value=900
)
bedrooms = st.sidebar.number_input(
    "Number of Bedrooms", min_value=1, max_value=10, value=2
)
full_bath = st.sidebar.number_input(
    "Full Bathrooms", min_value=0, max_value=5, value=2
)
half_bath = st.sidebar.number_input(
    "Half Bathrooms", min_value=0, max_value=5, value=1
)

# Prediction
if st.button("Predict Price"):
    features = np.array([[sqft, bedrooms, full_bath, half_bath]])
    prediction = model.predict(features)[0]

    st.success(f"💰 Estimated House Price: ₹ {prediction:,.2f}")

st.markdown("---")
st.caption("Built using Streamlit & Linear Regression")