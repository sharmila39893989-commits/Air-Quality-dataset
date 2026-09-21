import streamlit as st
import numpy as np
import pickle
import tensorflow as tf


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Air Quality ANN Prediction",
    page_icon="🌫️",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🌫️ Air Quality Prediction using ANN")

st.write(
    "TensorFlow ANN based Classification and Regression"
)


# --------------------------------------------------
# LOAD CLASSIFICATION MODEL
# --------------------------------------------------

classification_model = tf.keras.models.load_model(
    "classification_model.h5"
)


# --------------------------------------------------
# LOAD REGRESSION MODEL
# --------------------------------------------------

regression_model = tf.keras.models.load_model(
    "regression_model.h5"
)


# --------------------------------------------------
# LOAD CLASSIFICATION SCALER
# --------------------------------------------------

with open("classification_scaler.pkl", "rb") as file:
    classification_scaler = pickle.load(file)


# --------------------------------------------------
# LOAD ENCODER
# --------------------------------------------------

with open("classification_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)


# --------------------------------------------------
# LOAD REGRESSION SCALER
# --------------------------------------------------

with open("regression_scaler.pkl", "rb") as file:
    regression_scaler = pickle.load(file)


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("🌫️ Enter Air Quality Values")


col1, col2, col3 = st.columns(3)


with col1:

    pm25 = st.number_input(
        "PM2.5",
        min_value=0.0,
        value=50.0
    )

    pm10 = st.number_input(
        "PM10",
        min_value=0.0,
        value=80.0
    )


with col2:

    no2 = st.number_input(
        "NO2",
        min_value=0.0,
        value=30.0
    )

    so2 = st.number_input(
        "SO2",
        min_value=0.0,
        value=10.0
    )


with col3:

    co = st.number_input(
        "CO",
        min_value=0.0,
        value=1.0
    )

    o3 = st.number_input(
        "O3",
        min_value=0.0,
        value=40.0
    )


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button("🔮 Predict"):

    # ----------------------------------------------
    # CREATE INPUT ARRAY
    # ----------------------------------------------

    input_data = np.array([
        pm25,
        pm10,
        no2,
        so2,
        co,
        o3
    ]).reshape(1, -1)


    # ----------------------------------------------
    # CLASSIFICATION
    # ----------------------------------------------

    scaled_classification = classification_scaler.transform(
        input_data
    )


    class_prediction = classification_model.predict(
        scaled_classification,
        verbose=0
    )


    class_index = np.argmax(
        class_prediction,
        axis=1
    )[0]


    class_name = encoder.inverse_transform(
        [class_index]
    )[0]


    # ----------------------------------------------
    # DISPLAY CLASSIFICATION RESULT
    # ----------------------------------------------

    st.success(
        f"🌫️ Air Quality Category: {class_name}"
    )


    # ----------------------------------------------
    # REGRESSION
    # ----------------------------------------------

    scaled_regression = regression_scaler.transform(
        input_data
    )


    aqi_prediction = regression_model.predict(
        scaled_regression,
        verbose=0
    )


    # ----------------------------------------------
    # DISPLAY AQI RESULT
    # ----------------------------------------------

    st.info(
        f"📊 Predicted AQI: {aqi_prediction[0][0]:.2f}"
    )