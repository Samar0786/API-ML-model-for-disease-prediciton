import pandas as pd
from app.core.model_loader import get_model
from app.core.logger import logger

EXPECTED_COLUMNS = [
    "age_of_the_patient",
    "gender_of_the_patient",
    "total_bilirubin",
    "direct_bilirubin",
    "alkphos_alkaline_phosphotase",
    "sgpt_alamine_aminotransferase",
    "sgot_aspartate_aminotransferase",
    "total_protiens",
    "alb_albumin",
    "a/g_ratio_albumin_and_globulin_ratio",
]


def risk_category(prob):
    if prob < 0.35:
        return "LOW RISK"
    elif prob < 0.75:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"


def predict_liver(data: dict):
    try:
        logger.info("Liver prediction request")

        model = get_model("liver")

        # Fix key mismatch
        if "a_g_ratio_albumin_and_globulin_ratio" in data:
            data["a/g_ratio_albumin_and_globulin_ratio"] = data.pop(
                "a_g_ratio_albumin_and_globulin_ratio"
            )

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

    except Exception as e:
        logger.error(f"Liver error: {e}")
        raise