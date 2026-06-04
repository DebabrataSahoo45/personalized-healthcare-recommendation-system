# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:19:21 2026

@author: notif
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(
    "F:\\Amdox Technologies Internship\\personalised_dataset.csv"
)

# ==========================
# Drop Unnecessary Columns
# ==========================

df.drop(
    columns=[
        "Patient_ID",
        "Diet_Recommendation",
        "Exercise_Recommendation",
        "Predicted_Insurance_Cost"
    ],
    inplace=True
)

# ==========================
# Define Target Variable
# ==========================

target_column = "Health_Risk"

X = df.drop(columns=[target_column])
y = df[target_column]

# ==========================
# Encode Target Variable
# ==========================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

joblib.dump(
    label_encoder,
    "health_risk_encoder.pkl"
)

# ==========================
# Identify Feature Types
# ==========================

categorical_features = X.select_dtypes(
    include=["object"]
).columns

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns

# ==========================
# Preprocessing Pipelines
# ==========================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# ==========================
# Random Forest Model
# ==========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

# ==========================
# Create Pipeline
# ==========================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Train Model
# ==========================

pipeline.fit(
    X_train,
    y_train
)

# ==========================
# Predictions
# ==========================

predictions = pipeline.predict(
    X_test
)

# ==========================
# Evaluation
# ==========================

print("\nAccuracy\n")

print(
    accuracy_score(
        y_test,
        predictions
    )
)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

# ==========================
# Save Model
# ==========================

joblib.dump(
    pipeline,
    "health_risk_model.pkl"
)

print("\nModel Saved Successfully")