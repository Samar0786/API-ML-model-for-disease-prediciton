import pandas as pd
import os
from app.core.model_loader import load_model_by_name
from app.core.logger import logger

# Load model
model = load_model_by_name("diabetes")

# EXACT training columns (must match model.feature_names_in_)
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


def _build_smoking_history_map() -> dict[str, int]:
    """Recreate the same label encoding used during training (pd.factorize)."""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(base_dir, "data", "diabetes_prediction_dataset.csv")
        df = pd.read_csv(data_path).drop_duplicates()

        uniques = pd.factorize(df["smoking_history"])[1]
        return {str(value).strip().lower(): int(index) for index, value in enumerate(uniques)}
    except Exception as e:
        logger.warning(f"Could not build smoking_history map from dataset: {e}")
        return {
            "never": 0,
            "former": 1,
            "current": 2,
            "not current": 3,
            "ever": 4,
            "no info": 5,
        }


SMOKING_HISTORY_MAP = _build_smoking_history_map()


def _encode_gender(value: str | int | float) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"male", "m", "1"}:
        return 1
    if text in {"female", "f", "0"}:
        return 0
    raise ValueError(f"Invalid gender: {value}")


def _encode_smoking_history(value: str | int | float) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    key = str(value).strip().lower()
    if key in SMOKING_HISTORY_MAP:
        return SMOKING_HISTORY_MAP[key]
    raise ValueError(f"Invalid smoking_history: {value}")

# 🔥 Risk mapping
def risk_category(prob):
    if prob < 0.3:
        return "LOW RISK"
    elif prob < 0.7:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"


def predict_diabetes(data: dict):
    try:
        logger.info("Diabetes prediction request")

        # Accept either client key and normalize to trained feature name
        if "HbA1c_level" in data and "hba1c_level" not in data:
            data["hba1c_level"] = data.pop("HbA1c_level")

        # ----------------------------
        # VALIDATION
        # ----------------------------
        missing = [col for col in EXPECTED_COLUMNS if col not in data]
        if missing:
            raise ValueError(f"Missing fields: {missing}")

        # ----------------------------
        # TRANSFORM (IMPORTANT)
        # ----------------------------
        transformed = {
            "gender": _encode_gender(data["gender"]),
            "age": int(data["age"]),
            "hypertension": int(data["hypertension"]),
            "heart_disease": int(data["heart_disease"]),
            "smoking_history": _encode_smoking_history(data["smoking_history"]),
            "bmi": float(data["bmi"]),
            "hba1c_level": float(data["hba1c_level"]),
            "blood_glucose_level": float(data["blood_glucose_level"]),
        }

        df = pd.DataFrame([[transformed[col] for col in EXPECTED_COLUMNS]], columns=EXPECTED_COLUMNS)

        prob = model.predict_proba(df)[0][1]

        return {
            "prediction": int(prob >= 0.5),
            "probability": float(prob),
            "risk_level": risk_category(prob)
        }

    except Exception as e:
        logger.error(f"Diabetes prediction error: {e}")
        raise