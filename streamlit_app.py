import streamlit as st
import requests

# --- Configuration de la page ---
st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀")

st.title("Prédiction de Maladie Cardiaque")
st.markdown("Démo interactive du modèle MLOps — Cleveland Heart Disease Dataset")

# URL de l'API en production sur Render
API_URL = "https://heart-mlops-api.onrender.com/predict"

# --- Formulaire de saisie des données patient ---
st.header("Données du patient")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Âge", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sexe", options=[("Femme", 0), ("Homme", 1)], format_func=lambda x: x[0])
    cp = st.selectbox("Type de douleur thoracique", options=[
        ("Angine typique", 1), ("Angine atypique", 2),
        ("Douleur non-angineuse", 3), ("Asymptomatique", 4)
    ], format_func=lambda x: x[0])
    trestbps = st.number_input("Pression artérielle au repos (mmHg)", min_value=80, max_value=220, value=130)
    chol = st.number_input("Cholestérol (mg/dl)", min_value=100, max_value=600, value=220)
    fbs = st.selectbox("Glycémie à jeun > 120 mg/dl", options=[("Non", 0), ("Oui", 1)], format_func=lambda x: x[0])
    restecg = st.selectbox("ECG au repos", options=[
        ("Normal", 0), ("Anomalie ST-T", 1), ("Hypertrophie ventriculaire", 2)
    ], format_func=lambda x: x[0])

with col2:
    thalach = st.number_input("Fréquence cardiaque max", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Angine induite par l'effort", options=[("Non", 0), ("Oui", 1)], format_func=lambda x: x[0])
    oldpeak = st.number_input("Dépression ST (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    slope = st.selectbox("Pente du segment ST", options=[
        ("Montante", 1), ("Plate", 2), ("Descendante", 3)
    ], format_func=lambda x: x[0])
    ca = st.selectbox("Nombre de vaisseaux colorés", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassémie", options=[
        ("Normal", 3), ("Défaut fixe", 6), ("Défaut réversible", 7)
    ], format_func=lambda x: x[0])

# --- Bouton de prédiction ---
if st.button("Prédire", type="primary"):

    patient_data = {
        "age": age,
        "sex": sex[1],
        "cp": cp[1],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs[1],
        "restecg": restecg[1],
        "thalach": thalach,
        "exang": exang[1],
        "oldpeak": oldpeak,
        "slope": slope[1],
        "ca": ca,
        "thal": thal[1]
    }

    with st.spinner("Appel à l'API en cours..."):
        try:
            response = requests.post(API_URL, json=patient_data, timeout=30)
            result = response.json()

            st.divider()

            if result["prediction"] == 1:
                st.error(f"⚠️ {result['interpretation']}")
            else:
                st.success(f"✅ {result['interpretation']}")

            st.metric("Probabilité de maladie cardiaque", f"{result['probability']*100:.1f}%")

        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")

st.divider()
st.caption("Modèle : LogisticRegression (AUC 0.946) — Projet MLOps end-to-end")