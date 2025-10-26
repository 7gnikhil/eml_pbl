import streamlit as st
import requests
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="BMI Prediction Engine",
    page_icon="🧠",
    layout="centered"
)

# --- Backend API URL ---
# This is the address where your Flask backend (app.py) is running.
BACKEND_URL = "https://bmi-backend.onrender.com/predict"

# --- UI Components ---
st.title('🧠 BMI Prediction Engine')
st.write("Predict Body Mass Index (BMI) using either an image or numerical data.")

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Predict from Numerical Data")
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=1.0)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

    if st.button("Predict BMI from Data"):
        payload = {'height': height, 'weight': weight, 'age': age}
        try:
            response = requests.post(f"{BACKEND_URL}/predict_numerical", json=payload)
            response.raise_for_status()
            result = response.json()
            st.success(f"Predicted BMI: **{result['bmi']:.2f}** - Category: **{result['category']}**")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the backend. Is app.py running? Error: {e}")

with col2:
    st.subheader("Predict from an Image")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        with st.spinner('Analyzing image and predicting...'):
            buffered = io.BytesIO()
            # Convert to RGB if necessary to avoid JPEG save issues
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}

            try:
                response = requests.post(f"{BACKEND_URL}/predict_image", files=files)
                response.raise_for_status()
                result = response.json()
                st.success(f"Predicted BMI: **{result['bmi']:.2f}** - Category: **{result['category']}**")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the backend. Is app.py running? Error: {e}")

# --- Calorie Calculator Section ---
st.subheader("Calorie Calculator")
st.write("Calculate your Basal Metabolic Rate (BMR) and daily calorie needs based on activity level.")

col3, col4 = st.columns(2)

with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
    height_cal = st.number_input("Height (cm) for Calorie Calc", min_value=100.0, max_value=250.0, value=170.0, step=1.0)
    weight_cal = st.number_input("Weight (kg) for Calorie Calc", min_value=30.0, max_value=200.0, value=70.0, step=0.5)

with col4:
    age_cal = st.number_input("Age for Calorie Calc", min_value=18, max_value=100, value=30)
    activity = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"])

if st.button("Calculate Calories"):
    payload = {'gender': gender, 'height': height_cal, 'weight': weight_cal, 'age': age_cal, 'activity': activity}
    try:
        response = requests.post(f"{BACKEND_URL}/calorie_calculator", json=payload)
        response.raise_for_status()
        result = response.json()
        st.success(f"BMR: **{result['bmr']:.2f}** kcal/day\n\nDaily Calories Needed: **{result['daily_calories']:.2f}** kcal/day")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the backend. Is app.py running? Error: {e}")
