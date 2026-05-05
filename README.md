# Écho - Chatbot culturel
*Projet OpenClassrooms - Concevez et déployez un système RAG.*

> Écho est le chatbot culturel développé par Puls-Events. Il permet d’interroger une base d’événements OpenAgenda à l’aide d’un système RAG combinant recherche vectorielle FAISS et génération de réponse par Mistral.


## Objectif

Ce projet a pour objectif de concevoir un système RAG capable de répondre à des questions à partir de données collectées via OpenAgenda.

L'application s'appuiera notamment sur :

- Python 3.12
- Poetry pour la gestion de l'environnement
- LangChain pour la chaîne RAG
- Faiss pour l'index vectoriel
- Mistral AI pour le modèle de langage
- FastAPI pour l'exposition d'une API

## Installation

Installer les dépendances du projet avec Poetry :

```bash
poetry install --no-root
```

Vérifier la version de Python utilisée :

```bash
poetry run python --version
```

La version attendue est Python 3.12.

## Variables d'environnement

Créer un fichier `.env` à partir du fichier d'exemple :

```bash
cp .env.example .env
```

Puis renseigner les clés API nécessaires dans le fichier `.env`.

Le fichier `.env` ne doit jamais être versionné.

## Vérification de l'environnement

Tester que les principales dépendances sont correctement installées :

```bash
poetry run python -c "import pandas, requests, dotenv, pydantic, fastapi, langchain, faiss, mistralai; print('Imports OK')"
```

## Qualité du code

Lancer Ruff :

```bash
poetry run ruff check .
```

Lancer les tests :

```bash
poetry run pytest
```

