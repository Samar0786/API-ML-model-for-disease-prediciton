import joblib
import pandas as pd

# ----------------------------
# LOAD
# ----------------------------
model = joblib.load("models/kidney/kidney_model.pkl")
scaler = joblib.load("models/kidney/scaler.pkl")
imputer = joblib.load("models/kidney/imputer.pkl")
features = joblib.load("models/kidney/top_features.pkl")

# ----------------------------
# RISK FUNCTION
# ----------------------------
def risk_category(prob):
    if prob < 0.6:
        return "LOW RISK"
    elif prob < 0.85:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"

# ----------------------------
# FULL FEATURE INPUT (IMPORTANT)
# ----------------------------
# You MUST match full training feature structure

# Load training columns structure
dummy = pd.read_csv("data/Chronic_Kidney_Disease.csv")
dummy = dummy.drop(columns=["PatientID", "DoctorInCharge", "Diagnosis"], errors="ignore")

# Create empty row
input_df = pd.DataFrame([0]*len(dummy.columns)).T
input_df.columns = dummy.columns

# Fill ONLY required fields
input_values = {
    "SerumCreatinine": 1.0,
    "GFR": 90,
    "BUNLevels": 15,
    "ProteinInUrine": 0,
    "FastingBloodSugar": 180,
    "HbA1c": 8.5,
    "SystolicBP": 130,
    "HemoglobinLevels": 13
}

for key, value in input_values.items():
    input_df[key] = value

# ----------------------------
# PREPROCESS (CORRECT ORDER)
# ----------------------------
input_imputed = imputer.transform(input_df)
input_scaled = scaler.transform(input_imputed)

input_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)

# Select top features AFTER preprocessing
input_final = input_scaled[features]

# ----------------------------
# PREDICT
# ----------------------------
prob = model.predict_proba(input_final)[0][1]
risk = risk_category(prob)

# ----------------------------
# OUTPUT
# ----------------------------
print("\n===== TEST RESULT =====")
print("Probability:", round(prob, 3))
print("Risk Level:", risk)