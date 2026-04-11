import pandas as pd
from app.core.model_loader import get_model

EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]


def risk_category(prob):
    if prob < 0.4:
        return "LOW RISK"
    elif prob < 0.7:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"


def predict_heart(data: dict):
    model = get_model("heart")

    missing = [c for c in EXPECTED_COLUMNS if c not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    df = pd.DataFrame([[data[c] for c in EXPECTED_COLUMNS]],
                      columns=EXPECTED_COLUMNS)

    prob = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prob >= 0.5),
        "probability": float(prob),
        "risk_level": risk_category(prob)
    }