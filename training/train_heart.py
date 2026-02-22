import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier


#  Setup Paths (Production Safe)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models", "heart")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)



#  Load & Clean Dataset


data = pd.read_csv(DATA_PATH)

print("Original Shape:", data.shape)

data = data.drop_duplicates()

print("Shape After Removing Duplicates:", data.shape)
print("Duplicate Rows Removed:", data.duplicated().sum())

X = data.drop("target", axis=1)
y = data["target"].astype(int)

print("\nClass Distribution:")
print(y.value_counts())


#  Feature Groups


numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

numeric_features = [c for c in numeric_features if c in X.columns]
categorical_features = [c for c in categorical_features if c in X.columns]


#  Preprocessing Pipeline


numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", "passthrough", categorical_features)
    ]
)



#  Model Definition (Tuned XGBoost)

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", xgb_model)
])


#  Train/Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


#  Cross Validation


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

print("\nCross-Validation ROC-AUC: {:.4f} ± {:.4f}".format(
    cv_scores.mean(), cv_scores.std()
))


#  Train Final Model


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n===== Final Test Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#  SHAP Explainability


trained_xgb = model.named_steps["clf"]
X_processed = model.named_steps["preprocess"].transform(X)

explainer = shap.TreeExplainer(trained_xgb)
shap_values = explainer.shap_values(X_processed)

shap_plot_path = os.path.join(MODELS_DIR, "shap_feature_importance.png")

plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(shap_plot_path)
plt.close()

print("\nSHAP feature importance plot saved successfully!")


# Save Model


model_path = os.path.join(MODELS_DIR, "final_heart_model.pkl")
joblib.dump(model, model_path)

print("Model saved successfully!")



#  Feature Importance Table


feature_names = model.named_steps["preprocess"].get_feature_names_out()
importances = model.named_steps["clf"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n===== Feature Importance Ranking =====")
print(importance_df)

fi_path = os.path.join(MODELS_DIR, "feature_importance.csv")
importance_df.to_csv(fi_path, index=False)

print("Feature importance table saved successfully!")


#  Threshold Tuning


threshold = 0.40
y_custom_pred = (y_proba >= threshold).astype(int)

print(f"\n===== Custom Threshold Evaluation (Threshold = {threshold}) =====")
print("Accuracy:", accuracy_score(y_test, y_custom_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_custom_pred))
print("\nClassification Report:\n", classification_report(y_test, y_custom_pred))