import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_diesease_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction App by orange")
st.markdown("Provide the following details to predict the likelihood of heart disease:")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("sex",['M','F'])
chest_pain_type = st.selectbox("Chest Pain Type", ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
resting_blood_pressure = st.number_input("Resting Blood Pressure", min_value=80)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['True', 'False'])
rest_ecg = st.selectbox("Resting ECG", ['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'])
max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", ['Yes', 'No'])    
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=0.0)

if st.button('predict'):
    raw_data = {
        'age': age,
        'sex': 1 if sex == 'M' else 0,
        'chest_pain_type': chest_pain_type,
        'resting_blood_pressure': resting_blood_pressure,
        'cholesterol': cholesterol,
        'fasting_blood_sugar': 1 if fasting_blood_sugar == 'True' else 0,
        'rest_ecg': rest_ecg,
        'max_heart_rate': max_heart_rate,
        'exercise_induced_angina': 1 if exercise_induced_angina == 'Yes' else 0,
        'oldpeak': oldpeak

    }
    input_df = pd.DataFrame([raw_data])
    input_df = pd.get_dummies(input_df, columns=['chest_pain_type', 'rest_ecg'], drop_first=True)
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[columns]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.write(f'Predicted Heart Disease: {prediction}')
    if prediction[0] == 1:
        st.error("High risk of heart disease. Please consult a doctor fromm Orangeee.")
    else:
        st.success("Low risk of heart disease. Keep up the healthy lifestyle!")