# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:33:32 2026

@author: notif
"""
import pandas as pd
import joblib

# Load saved model
model = joblib.load("health_risk_model.pkl")

# Load label encoder
encoder = joblib.load("health_risk_encoder.pkl")

# Sample patient data
sample_patient = {
    "Age": 55,
    "Gender": "Male",
    "BMI": 28,
    "Smoking_Status": "Current smoker",
    "Alcohol_Consumption": "Low",
    "Physical_Activity_Level": "Lightly Active",
    "Diet_Type": "Vegetarian",
    "Blood_Pressure": "129/70",
    "Cholesterol": 210,
    "Glucose_Level": 95,
    "HbA1c": 5.5,
    "Heart_Disease_Risk": "Moderate",
    "Diabetes_Risk": "Low",
    "PRS_Cardiometabolic": 0.3,
    "PRS_Type2Diabetes": 0.2,
    "APOE_e4_Carrier": 1,
    "BRCA_Pathogenic_Variant": 0,
    "Family_History_CVD": 1,
    "Family_History_T2D": 0,
    "Stress_Level": 5,
    "Depression_Score": 3,
    "Anxiety_Score": 4,
    "Social_Isolation_Index": 2,
    "Sleep_Hours": 7,
    "Sleep_Quality": "Good",
    "Resting_Heart_Rate": 72,
    "HRV": 55,
    "Systolic_BP": 129,
    "Diastolic_BP": 70,
    "LDL": 110,
    "HDL": 55,
    "Triglycerides": 150,
    "CRP": 1.4,
    "eGFR": 92,
    "Waist_Circumference": 92
}

# Convert to DataFrame
patient_df = pd.DataFrame([sample_patient])

# Predict
prediction = model.predict(patient_df)

# Decode label
risk = encoder.inverse_transform(prediction)

print("\nPredicted Health Risk:")
print(risk[0])