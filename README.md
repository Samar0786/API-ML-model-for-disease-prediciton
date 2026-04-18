# 🏥 Smart Hospital AI – FastAPI ML Service

This repository contains the **Machine Learning API layer** for the Smart Hospital system.
It provides disease prediction endpoints powered by trained ML models.

---

## 🚀 Live API

👉 https://api-ml-model-for-disease-prediciton.onrender.com

---

## 🧠 Features

* ✅ Heart Disease Prediction
* ✅ Liver Disease Prediction
* ✅ Diabetes Prediction
* ✅ Risk Classification (LOW / MEDIUM / HIGH)
* ✅ SHAP Explainability (local training)
* ✅ FastAPI-based high-performance API
* ✅ Production deployed on Render

---

## ⚙️ Tech Stack

* Python 3.11
* FastAPI
* Scikit-learn
* XGBoost
* SHAP
* Uvicorn

---

## 📡 API Endpoints

### 🔹 Heart Disease

```http
POST /predict/heart
```

### 🔹 Liver Disease

```http
POST /predict/liver
```

### 🔹 Diabetes

```http
POST /predict/diabetes
```

---

## 📥 Example Request (Diabetes)

```json
{
  "gender": "Female",
  "age": 25,
  "hypertension": 0,
  "heart_disease": 0,
  "smoking_history": "never",
  "bmi": 22.0,
  "HbA1c_level": 5.2,
  "blood_glucose_level": 95
}
```

---

## 📤 Example Response

```json
{
  "prediction": 0,
  "probability": 0.384,
  "risk_level": "LOW RISK"
}
```

---

## 🏗️ Project Structure

```text
app/
│
├── main.py              # FastAPI entry point
├── routes/              # API endpoints
├── services/            # ML inference logic
├── core/                # model loader & utilities
│
models/
├── heart/
├── liver/
├── diabetes/
```

---

## 🧪 Run Locally

### 1. Clone repo

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run server

```bash
python -m uvicorn app.main:app --reload
```

---

## 📊 API Docs

After running locally:

👉 http://127.0.0.1:8000/docs

---

## ☁️ Deployment

Deployed on **Render**

* Runtime: Python
* Start Command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 10000
```

---

## ⚠️ Notes

* First request may take time (Render cold start)
* Models are preloaded at startup for performance
* Input schema must match training features

---

## 🔗 Integration

This API is consumed by:

* Node.js Backend (Hospital System)
* React Frontend (Dashboard UI)

---

## 📌 Future Improvements

* Add more disease models
* Add explainability endpoint
* Add model versioning
* Add authentication layer

---

## 👨‍💻 Author

Mohd Samar Bin Mahtab

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
