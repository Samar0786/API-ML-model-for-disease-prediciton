import pandas as pd
from app.core.model_loader import get_model
from app.core.logger import logger

EXPECTED_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "hba1c_level",
    "blood_glucose_level"
]


def risk_category(prob):
    if prob < 0.3:
        return "LOW RISK"
    elif prob < 0.7:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"


def _encode_gender(val):
    return 1 if str(val).lower() in ["male", "1"] else 0


def _encode_smoking(val):
    mapping = {
        "never": 0,
        "former": 1,
        "current": 2,
        "not current": 3,
        "ever": 4,
        "no info": 5,
    }
    return mapping.get(str(val).lower(), 0)


def predict_diabetes(data: dict):
    try:
        logger.info("Diabetes prediction request")

        model = get_model("diabetes")

        if "HbA1c_level" in data:
            data["hba1c_level"] = data.pop("HbA1c_level")

        missing = [c for c in EXPECTED_COLUMNS if c not in data]
        if missing:
            raise ValueError(f"Missing fields: {missing}")

        transformed = {
            "gender": _encode_gender(data["gender"]),
            "age": int(data["age"]),
            "hypertension": int(data["hypertension"]),
            "heart_disease": int(data["heart_disease"]),
            "smoking_history": _encode_smoking(data["smoking_history"]),
            "bmi": float(data["bmi"]),
            "hba1c_level": float(data["hba1c_level"]),
            "blood_glucose_level": float(data["blood_glucose_level"]),
        }

        df = pd.DataFrame([transformed])

        prob = model.predict_proba(df)[0][1]

        return {
            "prediction": int(prob >= 0.5),
            "probability": float(prob),
            "risk_level": risk_category(prob)
        }

    except Exception as e:
        logger.error(f"Diabetes error: {e}")
        raise