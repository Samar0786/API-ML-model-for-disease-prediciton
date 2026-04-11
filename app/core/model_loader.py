import os
import joblib

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
    "heart": "models/heart/final_heart_model.pkl",
    "diabetes": "models/diabetes/final_diabetes_model.pkl",
    "liver": "models/liver/liver_pipeline.pkl",
    # "kidney": "models/kidney/kidney_pipeline.pkl"
}

# ===============================
# CACHE (IMPORTANT)
# ===============================
models_cache = {}


# ===============================
# PRELOAD ALL MODELS (BEST PRACTICE)
# ===============================
def load_all_models():
    print("🚀 Loading ML models...")

    for name, relative_path in MODEL_FILES.items():
        full_path = os.path.join(BASE_DIR, relative_path)

        if not os.path.exists(full_path):
            print(f"⚠️ Model missing: {full_path}")
            continue

        try:
            models_cache[name] = joblib.load(full_path)
            print(f"✅ Loaded: {name}")
        except Exception as e:
            print(f"❌ Failed loading {name}: {e}")

    print("🎯 Model loading complete")


# ===============================
# GET MODEL (FAST ACCESS)
# ===============================
def get_model(name: str):
    name = name.lower()

    model = models_cache.get(name)

    if model is None:
        raise ValueError(f"Model '{name}' not loaded")

    return model


# ===============================
# OPTIONAL (DIRECT LOAD)
# ===============================
def load_model_by_path(relative_path: str):
    path = os.path.join(BASE_DIR, relative_path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)