import pandas as pd
from app.core.model_loader import load_model_by_name

# ===============================
# EXPECTED FEATURES
# ===============================
EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

# ===============================
# RISK FUNCTION (CONSISTENT)
# ===============================
def risk_category(prob):
    if prob < 0.4:
        return "LOW RISK"
    elif prob < 0.7:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_heart(data: dict):

    # Load model (correct loader)
    model = load_model_by_name("heart")

    #  Check missing fields (INSIDE function)
    missing = [col for col in EXPECTED_COLUMNS if col not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    # Ensure correct column order
    df = pd.DataFrame(
        [[data[col] for col in EXPECTED_COLUMNS]],
        columns=EXPECTED_COLUMNS
    )

    #  Prediction
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(proba),
        "risk_level": risk_category(proba)
    }