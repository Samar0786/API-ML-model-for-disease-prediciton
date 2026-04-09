import os
import joblib
from fastapi import HTTPException

# ===============================
# BASE DIRECTORY
# ===============================
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# ===============================
# MODEL REGISTRY
# ===============================
MODEL_FILES = {
    "heart": "final_heart_model.pkl",
    "diabetes": "final_diabetes_model.pkl",
    "liver": "liver_pipeline.pkl",
    # "kidney": "kidney_pipeline.pkl"  # add later
}

# ===============================
# CACHE (IMPORTANT FOR PERFORMANCE)
# ===============================
models_cache = {}

# ===============================
# LOAD MODEL BY DISEASE (API USE)
# ===============================
def load_model_by_name(disease: str):
    disease = disease.lower()

    if disease in models_cache:
        return models_cache[disease]

    if disease not in MODEL_FILES:
        raise HTTPException(status_code=404, detail=f"{disease} model not supported")

    model_path = os.path.join(
        BASE_DIR,
        "models",
        disease,
        MODEL_FILES[disease]
    )

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=500,
            detail=f"{disease} model not found at {model_path}"
        )

    model = joblib.load(model_path)
    models_cache[disease] = model

    return model


# ===============================
# LOAD MODEL BY PATH (OPTIONAL USE)
# ===============================
def load_model_by_path(relative_path: str):
    path = os.path.join(BASE_DIR, relative_path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)