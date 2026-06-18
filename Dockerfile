# Image de base — Python 3.12 version légère
FROM python:3.12-slim

# Dossier de travail dans le container
WORKDIR /app

# Copier et installer les dépendances de l'API
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copier le code source et le modèle entraîné
COPY src/ ./src/
COPY mlruns/ ./mlruns/

# Exposer le port utilisé par l'API
EXPOSE 8000

# Lancer l'API avec uvicorn, accessible depuis l'extérieur du container
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]