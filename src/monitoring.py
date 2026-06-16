import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset
from src.preprocessing import preprocess

# --- Chargement des données de référence (entraînement) ---
def load_reference_data():
    """Charge et prépare les données d'entraînement comme référence."""
    df = preprocess('data/heart.csv')
    return df

# --- Simulation de nouvelles données en production ---
def simulate_production_data(df_reference):
    """
    Simule un batch de nouvelles données avec drift.
    En production réelle, ce seraient de vrais nouveaux patients.
    """
    df_production = df_reference.copy()
    n = len(df_production)

    # Simuler un drift sur les variables les plus à risque
    df_production['age'] = df_production['age'] + np.random.normal(5, 2, n)
    df_production['thalach'] = df_production['thalach'] - np.random.normal(10, 3, n)
    df_production['oldpeak'] = df_production['oldpeak'] + np.random.normal(0.5, 0.2, n)
    df_production['chol'] = df_production['chol'] + np.random.normal(15, 5, n)

    return df_production

if __name__ == "__main__":

    # --- Charger les données ---
    print("Chargement des données de référence...")
    df_reference = load_reference_data()

    print("Simulation des données de production...")
    df_production = simulate_production_data(df_reference)

    # --- Définir le schéma des données ---
    definition = DataDefinition()

    # --- Créer les datasets Evidently ---
    reference = Dataset.from_pandas(df_reference, data_definition=definition)
    current = Dataset.from_pandas(df_production, data_definition=definition)

    # --- Générer le rapport de drift ---
    print("Génération du rapport Evidently...")
    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference, current)
    my_eval.save_html("monitoring/drift_report.html")

    print("✅ Rapport généré : monitoring/drift_report.html")