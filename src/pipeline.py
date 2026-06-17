import sys
sys.path.append('.')

import subprocess
from prefect import flow, task

# --- Task 1 : Préprocessing ---
@task(name="preprocess-data", retries=1)
def preprocess_data():
    """Lance le script de preprocessing."""
    print("Étape 1 : Préprocessing des données...")
    result = subprocess.run(
        [sys.executable, "src/preprocessing.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Preprocessing échoué : {result.stderr}")
    return "preprocessing_ok"

# --- Task 2 : Entraînement ---
@task(name="train-models", retries=1)
def train_models():
    """Lance le script d'entraînement avec tracking MLflow."""
    print("Étape 2 : Entraînement des modèles...")
    result = subprocess.run(
        [sys.executable, "src/train.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Entraînement échoué : {result.stderr}")
    return "training_ok"

# --- Task 3 : Vérification du drift ---
@task(name="check-drift", retries=1)
def check_drift():
    """Lance le monitoring Evidently et vérifie le drift."""
    print("Étape 3 : Vérification du drift...")
    result = subprocess.run(
        [sys.executable, "src/monitoring.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Monitoring échoué : {result.stderr}")
    return "monitoring_ok"

# --- Flow principal ---
@flow(name="heart-disease-pipeline")
def ml_pipeline():
    """Pipeline complet : preprocessing → entraînement → monitoring."""

    # Étape 1
    preprocess_status = preprocess_data()
    print(f"✅ {preprocess_status}")

    # Étape 2
    train_status = train_models()
    print(f"✅ {train_status}")

    # Étape 3
    drift_status = check_drift()
    print(f"✅ {drift_status}")

    print("🎉 Pipeline terminé avec succès !")

if __name__ == "__main__":
    ml_pipeline()