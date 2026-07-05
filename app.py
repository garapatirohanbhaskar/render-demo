import streamlit as st
import pickle
import joblib
import pandas as pd

# Load files
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

st.title("❤️ Heart Disease Prediction")

# Input fields
age = st.number_input("Age", 1, 100)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting BP", 0, 300)
cholesterol = st.number_input("Cholesterol", 0, 600)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max HR", 50, 250)
exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
oldpeak = st.number_input("Oldpeak", value=0.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")