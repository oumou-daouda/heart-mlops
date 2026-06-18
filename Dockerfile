# Image de base — Python 3.12 version légère (slim)
FROM python:3.12-slim

# Dossier de travail dans le container
WORKDIR /app

# Copier et installer les dépendances en premier
# (optimisation : Docker met cette couche en cache si requirements-api.txt ne change pas)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copier le code source et le modèle exporté (fichier .pkl autonome)
COPY src/ ./src/
COPY models/ ./models/

# Exposer le port utilisé par l'API
EXPOSE 8000

# Commande lancée quand le container démarre
# --host 0.0.0.0 : écoute sur toutes les interfaces réseau (nécessaire pour le cloud)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]