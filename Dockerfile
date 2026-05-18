# Dockerfile pour l'API FastAPI d'Écho (POC RAG).
#
# Image volontairement simple, niveau étudiant :
# - base Python 3.12 slim
# - Poetry pour installer les dépendances du projet (groupe main uniquement)
# - copie du code, de l'index FAISS et des documents préparés
# - lancement Uvicorn sur le port 8000
#
# L'index FAISS doit déjà être construit localement avant le build :
# `poetry run python scripts/rebuild_index.py`
# Aucun secret n'est embarqué : la clé MISTRAL_API_KEY est passée au
# runtime via docker-compose.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.3.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# Poetry est installé via pip pour rester simple et reproductible.
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Installation des dépendances applicatives uniquement (pas les outils de dev).
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --no-root --only main

# Code applicatif.
COPY echo_app ./echo_app
COPY src ./src

# Index FAISS pré-construit localement (chargé au premier /ask).
COPY vector_store ./vector_store

# Documents préparés (utilisés uniquement si /rebuild est appelé).
COPY data/processed ./data/processed

EXPOSE 8000

CMD ["uvicorn", "echo_app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
