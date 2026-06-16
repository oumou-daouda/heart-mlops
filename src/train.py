import sys
sys.path.append('.')
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.preprocessing import preprocess

# --- Chargement et préparation des données ---
def load_data():
    df = preprocess('data/heart.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# --- Calcul des métriques ---
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba)
    }

# --- Entraînement avec tracking MLflow ---
def train_model(model, model_name, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):

        # Logger les hyperparamètres
        mlflow.log_params(params)

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Logger les métriques
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Sauvegarder le modèle comme artifact
        mlflow.sklearn.log_model(model, "model")

        print(f"{model_name} → AUC: {metrics['auc']:.3f} | F1: {metrics['f1']:.3f} | Accuracy: {metrics['accuracy']:.3f}")

if __name__ == "__main__":

    # --- Chargement des données ---
    X_train, X_test, y_train, y_test = load_data()

    # --- Définition de l'experiment MLflow ---
    mlflow.set_experiment("heart-disease-classification")

    # --- Modèle 1 : Régression Logistique ---
    # On applique log1p sur oldpeak (distribution asymétrique)
    X_train_lr = X_train.copy()
    X_test_lr = X_test.copy()
    X_train_lr['oldpeak'] = np.log1p(X_train_lr['oldpeak'])
    X_test_lr['oldpeak'] = np.log1p(X_test_lr['oldpeak'])

    params_lr = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
    train_model(
        LogisticRegression(**params_lr),
        "LogisticRegression",
        params_lr,
        X_train_lr, X_test_lr, y_train, y_test
    )

    # --- Modèle 2 : Random Forest ---
    params_rf = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    train_model(
        RandomForestClassifier(**params_rf),
        "RandomForest",
        params_rf,
        X_train, X_test, y_train, y_test
    )

    # --- Modèle 3 : XGBoost ---
    params_xgb = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42}
    train_model(
        XGBClassifier(**params_xgb),
        "XGBoost",
        params_xgb,
        X_train, X_test, y_train, y_test
    )