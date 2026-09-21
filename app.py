import streamlit as st
import numpy as np
import pickle
import tensorflow as tf


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Air Quality ANN Prediction",
    page_icon="🌫️",
    layout="wide"
)


# ==================================================
# TITLE
# ==================================================

st.title("🌫️ Air Quality Prediction using ANN")
st.subheader("TensorFlow ANN - Classification + Regression")

st.write(
    "Enter the air pollutant values to predict "
    "Air Quality Category and AQI."
)

st.divider()


# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_models():

    classification_model = tf.keras.models.load_model(
        "classification_model.h5",
        compile=False
    )

    regression_model = tf.keras.models.load_model(
        "regression_model.h5",
        compile=False
    )

    return classification_model, regression_model


classification_model, regression_model = load_models()


# ==================================================
# LOAD SCALERS
# ==================================================

with open("classification_scaler.pkl", "rb") as file:
    classification_scaler = pickle.load(file)


with open("regression_scaler.pkl", "rb") as file:
    regression_scaler = pickle.load(file)


# ==================================================
# LOAD ENCODER
# ==================================================

with open("classification_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)


# ==================================================
# INPUT SECTION
# ==================================================

st.header("🌍 Air Quality Input Values")


col1, col2 = st.columns(2)


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

    no = st.number_input(
        "NO",
        min_value=0.0,
        value=10.0
    )

    no2 = st.number_input(
        "NO2",
        min_value=0.0,
        value=20.0
    )


with col2:

    nox = st.number_input(
        "NOx",
        min_value=0.0,
        value=30.0
    )

    nh3 = st.number_input(
        "NH3",
        min_value=0.0,
        value=20.0
    )

    co = st.number_input(
        "CO",
        min_value=0.0,
        value=1.0
    )

    so2 = st.number_input(
        "SO2",
        min_value=0.0,
        value=10.0
    )


# ==================================================
# PREDICTION BUTTON
# ==================================================

st.divider()


if st.button(
    "🔮 Predict Air Quality",
    use_container_width=True
):

    # ==================================================
    # CREATE 8-FEATURE INPUT
    # ==================================================

    input_data = np.array([
        pm25,
        pm10,
        no,
        no2,
        nox,
        nh3,
        co,
        so2
    ], dtype=float).reshape(1, -1)


    # ==================================================
    # CLASSIFICATION
    # ==================================================

    try:

        scaled_classification = (
            classification_scaler.transform(
                input_data
            )
        )

        classification_prediction = (
            classification_model.predict(
                scaled_classification,
                verbose=0
            )
        )

        class_index = np.argmax(
            classification_prediction,
            axis=1
        )[0]

        class_name = encoder.inverse_transform(
            [class_index]
        )[0]

        st.success(
            f"🌫️ Air Quality Category: {class_name}"
        )

    except Exception as e:

        st.error(
            f"Classification Error: {e}"
        )


    # ==================================================
    # REGRESSION
    # ==================================================

    try:

        scaled_regression = (
            regression_scaler.transform(
                input_data
            )
        )

        aqi_prediction = (
            regression_model.predict(
                scaled_regression,
                verbose=0
            )
        )

        predicted_aqi = float(
            aqi_prediction[0][0]
        )

        st.info(
            f"📊 Predicted AQI: {predicted_aqi:.2f}"
        )

    except Exception as e:

        st.error(
            f"Regression Error: {e}"
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Built using Python, TensorFlow, Keras, "
    "Scikit-learn and Streamlit"
)
