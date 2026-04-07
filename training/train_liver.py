import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from xgboost import XGBClassifier

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Liver_disease.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "liver")

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH, encoding='latin-1')

print("Original Shape:", df.shape)

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)

# Remove duplicates
df = df.drop_duplicates()

print("Shape After Cleaning:", df.shape)

# ===============================
# TARGET
# ===============================
target_col = "result"

# Map: 1 = disease, 2 = no disease → 0
df[target_col] = df[target_col].replace({2: 0})

X = df.drop(target_col, axis=1)
y = df[target_col].astype(int)

print("\nClass Distribution:")
print(y.value_counts())

# ===============================
# FEATURE TYPES
# ===============================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric:", numeric_features)
print("Categorical:", categorical_features)

# ===============================
# PREPROCESSOR
# ===============================
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# HANDLE IMBALANCE
# ===============================
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# ===============================
# PIPELINE (FINAL MODEL)
# ===============================
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", CalibratedClassifierCV(
        XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            gamma=0.2,
            reg_lambda=1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),
        method="sigmoid",
        cv=5
    ))
])

# ===============================
# TRAIN
# ===============================
pipeline.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
print("Cross-validated ROC-AUC:", cv_score.mean())
# ===============================
# EVALUATION
# ===============================
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# SAVE PIPELINE
# ===============================
joblib.dump(pipeline, os.path.join(MODEL_DIR, "liver_pipeline.pkl"))

print("\nPipeline saved!")

# ===============================
# SHAP EXPLAINABILITY
# ===============================
print("\nGenerating SHAP...")

# Extract trained XGB model
calibrated = pipeline.named_steps["model"]
xgb_model = calibrated.calibrated_classifiers_[0].estimator

# Transform data
X_processed = pipeline.named_steps["preprocessor"].transform(X)

# Feature names after encoding
cat_features = pipeline.named_steps["preprocessor"] \
    .named_transformers_["cat"]["onehot"] \
    .get_feature_names_out(categorical_features)

feature_names = numeric_features + list(cat_features)

X_processed = pd.DataFrame(X_processed, columns=feature_names)

# SHAP
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_processed)

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_processed, plot_type="bar", show=False)
plt.savefig(os.path.join(MODEL_DIR, "shap_liver.png"))
plt.close()

print("SHAP saved!")