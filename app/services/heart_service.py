import pandas as pd
from app.core.model_loader import load_model

EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

def predict_heart(data: dict):

    model = load_model("heart")

    # Create dataframe with fixed column order
    df = pd.DataFrame([[data[col] for col in EXPECTED_COLUMNS]],
                      columns=EXPECTED_COLUMNS)

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    # Risk thresholds
    if proba >= 0.7:
        risk = "High"
    elif proba >= 0.4:
        risk = "Moderate"
    else:
        risk = "Low"

    return {
        "prediction": int(pred),
        "probability": float(proba),
        "risk_level": risk
    }