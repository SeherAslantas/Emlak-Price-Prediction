
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("emlak.pkl")

model = load_model()

st.title("🏠 House Price Prediction")
st.write("Predict house price per unit area using property details.")

st.divider()

st.subheader("Enter House Information")

house_age = st.number_input("House Age (years)", min_value=0.0, value=10.0)

distance = st.number_input(
    "Distance to the nearest MRT station",
    min_value=0.0,
    value=1000.0
)

stores = st.number_input(
    "Number of Convenience Stores",
    min_value=0,
    value=5
)

latitude = st.number_input("Latitude", value=24.98, format="%.6f")

longitude = st.number_input("Longitude", value=121.54, format="%.6f")

year = st.number_input(
    "Transaction Year",
    min_value=2000,
    max_value=2030,
    value=2013
)

month = st.selectbox("Transaction Month", range(1, 13))

if st.button("Predict House Price", type="primary"):

    data = pd.DataFrame([{
        "House age": house_age,
        "Distance to the nearest MRT station": distance,
        "Number of convenience stores": stores,
        "Latitude": latitude,
        "Longitude": longitude,
        "Transaction_year": year,
        "Transaction_month": month
    }])

    prediction = model.predict(data)[0]

    st.success(f"Predicted House Price: {prediction:.2f}")

    st.info("The result represents the predicted house price per unit area.")
