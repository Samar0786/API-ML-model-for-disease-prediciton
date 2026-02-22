import os
import joblib
from fastapi import HTTPException

# Correct project root (go up TWO levels from app/core/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

MODEL_FILES = {
    "heart": "final_heart_model.pkl",
    "diabetes": "final_diabetes_model.pkl",
}

models_cache = {}

def load_model(disease: str):

    if disease in models_cache:
        return models_cache[disease]

    if disease not in MODEL_FILES:
        raise HTTPException(status_code=404, detail="Model not supported")

    model_path = os.path.join(
        BASE_DIR,
        "models",
        disease,
        MODEL_FILES[disease]
    )

    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail=f"{disease} model file not found at {model_path}")

    model = joblib.load(model_path)
    models_cache[disease] = model
    return model