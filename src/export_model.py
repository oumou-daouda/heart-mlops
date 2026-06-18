import sys
sys.path.append('.')

import mlflow
import mlflow.sklearn
import joblib

# --- Connexion à MLflow ---
mlflow.set_tracking_uri("http://localhost:5000")

# --- Chargement du modèle depuis le Model Registry ---
print("Chargement de HeartModel v1 depuis MLflow...")
model = mlflow.sklearn.load_model("models:/HeartModel/1")

# --- Export en fichier autonome ---
joblib.dump(model, "models/heart_model.pkl")
print("Modele exporte : models/heart_model.pkl")