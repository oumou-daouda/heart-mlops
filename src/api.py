import sys
sys.path.append('.')

import mlflow
import mlflow.sklearn
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

# --- Chargement du modèle au démarrage de l'API ---
mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.sklearn.load_model("models:/HeartModel/1")

# --- Endpoint GET — vérifier que l'API tourne ---
@app.get("/health")
def health():
    return {"status": "ok", "model": "HeartModel v1"}

# --- Endpoint POST — faire une prédiction ---
@app.post("/predict")
def predict(data: PatientData):

    # Convertir les données en tableau numpy
    features = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        np.log1p(data.oldpeak),  # transformation log1p comme à l'entraînement
        data.slope, data.ca, data.thal
    ]])

    # Prédiction et probabilité
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 3),
        "interpretation": "Maladie cardiaque détectée" if prediction == 1 else "Pas de maladie cardiaque"
    }