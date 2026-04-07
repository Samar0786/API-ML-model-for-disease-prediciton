import os
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from xgboost import XGBClassifier


# ===============================
# Paths
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes_prediction_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "diabetes")

os.makedirs(MODEL_DIR, exist_ok=True)


# ===============================
# Load Dataset
# ===============================

data = pd.read_csv(DATA_PATH)

print("Original Shape:", data.shape)

# Remove duplicates
data = data.drop_duplicates()

print("Shape After Removing Duplicates:", data.shape)

print("\nColumns:", data.columns)


# ===============================
# Rename Columns (important)
# ===============================

data.columns = data.columns.str.lower()


# ===============================
# Encode categorical columns
# ===============================

# gender: Female=0, Male=1
data["gender"] = data["gender"].map({"Female": 0, "Male": 1})

# smoking_history → label encoding
data["smoking_history"] = pd.factorize(data["smoking_history"])[0]


# ===============================
# Features & Target
# ===============================

X = data.drop("diabetes", axis=1)
y = data["diabetes"]


print("\nClass Distribution:")
print(y.value_counts())


# ===============================
# Compute class imbalance weight
# ===============================

neg = (y == 0).sum()
pos = (y == 1).sum()

scale_pos_weight = neg / pos

print("\nscale_pos_weight:", scale_pos_weight)


# ===============================
# Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ===============================
# Model
# ===============================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.2,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.2,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42
    ))
])


# ===============================
# Cross Validation
# ===============================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

print("\nCross Validation ROC-AUC:", scores.mean())


# ===============================
# Train Model
# ===============================

model.fit(X_train, y_train)


# ===============================
# Predictions
# ===============================

y_proba = model.predict_proba(X_test)[:, 1]

# threshold tuning
threshold = 0.55
y_pred = (y_proba >= threshold).astype(int)


print("\n===== Test Results =====")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ===============================
# SHAP Explainability
# ===============================

print("\nGenerating SHAP explainability...")

xgb_model = model.named_steps["clf"]

X_scaled = model.named_steps["scaler"].transform(X)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_scaled)

plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)

shap_path = os.path.join(MODEL_DIR, "shap_feature_importance.png")
plt.savefig(shap_path)
plt.close()

print("SHAP plot saved at:", shap_path)


# ===============================
# Feature Importance
# ===============================

importances = xgb_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n===== Feature Importance =====")
print(importance_df)

importance_df.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)


# ===============================
# Save Model
# ===============================

model_path = os.path.join(MODEL_DIR, "final_diabetes_model.pkl")

joblib.dump(model, model_path)

print("\nModel saved at:", model_path)