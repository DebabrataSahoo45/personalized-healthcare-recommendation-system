# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:37:12 2026

@author: notif
"""

import pandas as pd

# Load dataset
df = pd.read_csv(
    "personalised_dataset.csv"
)

def recommend_healthcare(
        age,
        bmi,
        health_risk):

    filtered = df[
        df["Health_Risk"] == health_risk
    ].copy()

    filtered["distance"] = (
        abs(filtered["Age"] - age)
        +
        abs(filtered["BMI"] - bmi)
    )

    best_match = filtered.sort_values(
        by="distance"
    ).iloc[0]

    return {
        "Diet": best_match[
            "Diet_Recommendation"
        ],
        "Exercise": best_match[
            "Exercise_Recommendation"
        ]
    }

result = recommend_healthcare(
    age=55,
    bmi=28,
    health_risk="Moderate"
)

print("\nRecommended Diet:")
print(result["Diet"])

print("\nRecommended Exercise:")
print(result["Exercise"])