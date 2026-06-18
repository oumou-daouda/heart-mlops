import sys
sys.path.append('.')

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# --- Initialisation de l'application FastAPI ---
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API de prédiction de maladie cardiaque — Cleveland Heart Disease Dataset",
    version="1.0.0"
)

# --- Schéma des données attendues (validation automatique) ---
class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# --- Chargement du modèle exporté (fichier autonome, pas besoin de MLflow) ---
model = joblib.load("models/heart_model.pkl")

# --- Endpoint GET — vérifier que l'API tourne ---
@app.get("/health")
def health():
    return {"status": "ok", "model": "HeartModel v1"}

# --- Endpoint POST — faire une prédiction ---
@app.post("/predict")
def predict(data: PatientData):

    features = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        np.log1p(data.oldpeak),
        data.slope, data.ca, data.thal
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 3),
        "interpretation": "Maladie cardiaque détectée" if prediction == 1 else "Pas de maladie cardiaque"
    }