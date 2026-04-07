import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from xgboost import XGBClassifier

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Chronic_Kidney_Disease.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "kidney")

os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
data = pd.read_csv(DATA_PATH)
print("Original shape:", data.shape)



# ----------------------------
# CLEAN DATA
# ----------------------------
data = data.drop(columns=["PatientID", "DoctorInCharge"], errors="ignore")

# ----------------------------
# TARGET
# ----------------------------
target_col = "Diagnosis"

if data[target_col].dtype == "object":
    data[target_col] = data[target_col].map({
        "CKD": 1,
        "No CKD": 0,
        "Yes": 1,
        "No": 0,
        "Positive": 1,
        "Negative": 0
    })

X = data.drop(target_col, axis=1)
y = data[target_col]

print("\nClass Distribution:")
print(y.value_counts())
# ----------------------------
# ENCODE CATEGORICAL
# ----------------------------
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# ----------------------------
# SELECT CLINICAL FEATURES
# ----------------------------
selected_features = [
    'SerumCreatinine',
    'GFR',
    'BUNLevels',
    'ProteinInUrine',
    'FastingBloodSugar',
    'HbA1c',
    'SystolicBP',
    'HemoglobinLevels'
]

X = X[selected_features]

print("\nUsing Features:\n", selected_features)

# ----------------------------
# TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ----------------------------
# HANDLE IMBALANCE
# ----------------------------
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# ----------------------------
# BUILD PIPELINE
# ----------------------------
pipeline = Pipeline(steps=[
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler", StandardScaler()),
    ("model", CalibratedClassifierCV(
        XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            gamma=1,
            reg_lambda=2,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),
        method="sigmoid",
        cv=5
    ))
])

# ----------------------------
# TRAIN
# ----------------------------
pipeline.fit(X_train, y_train)

# ----------------------------
# EVALUATE
# ----------------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# SAVE MODEL
# ----------------------------
joblib.dump(pipeline, os.path.join(MODEL_DIR, "kidney_pipeline.pkl"))
joblib.dump(selected_features, os.path.join(MODEL_DIR, "top_features.pkl"))

print("\nPipeline saved successfully!")

# ----------------------------
# SHAP EXPLAINABILITY
# ----------------------------
print("\nGenerating SHAP...")

# Step 1: Get trained calibrated model
calibrated_model = pipeline.named_steps["model"]

# Step 2: Extract underlying XGBoost model
xgb_model = calibrated_model.calibrated_classifiers_[0].estimator

# Step 3: Apply preprocessing BEFORE SHAP
X_processed = pipeline.named_steps["imputer"].transform(X)
X_processed = pipeline.named_steps["scaler"].transform(X_processed)

# Convert back to DataFrame
X_processed = pd.DataFrame(X_processed, columns=X.columns)

# Step 4: SHAP
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_processed)

# Step 5: Plot
plt.figure()
shap.summary_plot(shap_values, X_processed, plot_type="bar", show=False)
plt.savefig(os.path.join(MODEL_DIR, "shap_feature_importance.png"))
plt.close()

print("SHAP plot saved!")