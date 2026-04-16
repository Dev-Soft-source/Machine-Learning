import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.predict import predict_house_value


st.set_page_config(page_title="Housing Price Prediction", layout="centered")

st.title("Housing Price Prediction")
st.write("Enter housing information to predict the median house value.")

med_inc = st.number_input("Median Income", min_value=0.0, value=3.5)
house_age = st.number_input("House Age", min_value=1.0, value=25.0)
ave_rooms = st.number_input("Average Rooms", min_value=0.1, value=5.5)
ave_bedrms = st.number_input("Average Bedrooms", min_value=0.1, value=1.1)
population = st.number_input("Population", min_value=1.0, value=1000.0)
ave_occup = st.number_input("Average Occupancy", min_value=0.1, value=3.0)
latitude = st.number_input("Latitude", value=34.05)
longitude = st.number_input("Longitude", value=-118.24)

if st.button("Predict Price"):
    sample = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }

    prediction = predict_house_value(sample)
    st.success(f"Predicted House Value: ${prediction * 100000:,.2f}")