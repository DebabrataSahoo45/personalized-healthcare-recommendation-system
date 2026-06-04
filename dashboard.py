# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:41:38 2026

@author: notif
"""

import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("F:\\Amdox Technologies Internship\\personalised_dataset.csv")


st.title(
    "Personalized Healthcare Recommendation System"
)

st.subheader(
    "Dataset Overview"
)

st.write(df.head())

st.subheader(
    "Health Risk Distribution"
)

fig1 = px.histogram(
    df,
    x="Health_Risk"
)

st.plotly_chart(fig1)

st.subheader(
    "Diabetes Risk Distribution"
)

fig2 = px.histogram(
    df,
    x="Diabetes_Risk"
)

st.plotly_chart(fig2)

st.subheader(
    "BMI vs Glucose Level"
)

fig3 = px.scatter(
    df,
    x="BMI",
    y="Glucose_Level",
    color="Health_Risk"
)

st.plotly_chart(fig3)