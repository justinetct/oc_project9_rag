# Écho - Chatbot culturel
*Projet OpenClassrooms - Concevez et déployez un système RAG.*

> Écho est le chatbot culturel développé par Puls-Events. Il permet d’interroger une base d’événements OpenAgenda à l’aide d’un système RAG combinant recherche vectorielle FAISS et génération de réponse par Mistral.

## Objectif

Concevoir un système RAG capable de répondre à des questions à partir d’événements collectés via OpenAgenda.

Les données sont récupérées depuis l’API Opendatasoft / OpenAgenda, puis pré-traitées sous forme de documents Markdown indexables. Ces documents seront ensuite découpés, indexés avec FAISS et utilisés par le chatbot Écho pour générer des réponses contextualisées.

## Sommaire

- [Objectif](#objectif)
- [Stack technique](#stack-technique)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Commandes utiles](#commandes-utiles)
  - [Qualité du code](#qualité-du-code)
  - [Pipeline OpenAgenda](#pipeline-openagenda)

## Stack technique

L'application s'appuiera notamment sur :

- Python 3.12
- Poetry pour la gestion de l'environnement
- LangChain pour la chaîne RAG
- FAISS pour l'index vectoriel
- Mistral AI pour le modèle de langage
- FastAPI pour l'exposition d'une API

Le **rapport technique** du projet sera rédigé dans [`docs/rapport_technique.md`](docs/rapport_technique.md).

La documentation du pipeline OpenAgenda, du pré-processing et de la construction des documents textuels est détaillée dans [`docs/openagenda_preprocessing.md`](docs/openagenda_preprocessing.md).

## Structure du projet

```text
oc_project9_rag/
├── .env.example                       # Exemple de variables d'environnement
├── data/                              # Données du projet
│   ├── evaluation/                    # Jeux de données utilisés pour évaluer le RAG
│   ├── processed/                     # Données nettoyées ou transformées
│   └── raw/                           # Données brutes collectées depuis OpenAgenda
├── docs/                              # Documentation projet et notes de démonstration
├── echo_app/                          # Application principale Écho
│   ├── api/main.py                    # Point d'entrée FastAPI
│   ├── config.py                      # Configuration spécifique à l'application
│   ├── indexing/                      # Création et mise à jour de l'index vectoriel
│   └── rag/                           # Logique RAG : recherche, prompt et génération
├── notebooks/                         # Notebooks d'exploration et d'analyse
├── pyproject.toml                     # Configuration Poetry
├── README.md
├── scripts/                           # Scripts exécutables ponctuels
│   ├── 01_fetch_openagenda_events.py  # Récupération des événements OpenAgenda bruts
│   ├── 02_filter_openagenda_events.py # Filtrage temporel des événements
│   ├── 03_clean_openagenda_events.py  # Nettoyage et normalisation des événements
│   └── 04_build_event_documents.py    # Création des documents textuels pour le RAG
├── src/                               # Code commun et fonctions utilitaires
│   ├── config.py                      # Constantes, chemins et paramètres de collecte
│   ├── documents.py                   # Construction des documents textuels indexables
│   ├── openagenda.py                  # Client simple pour l'API OpenAgenda
│   ├── preprocessing.py               # Filtrage, nettoyage et normalisation des événements
│   └── utils/io.py                    # Fonctions simples d'entrée / sortie
├── tests/                             # Tests automatisés
└── vector_store/                      # Index FAISS généré localement
```

Les dossiers `data/` et `vector_store/` contiennent des fichiers générés ou volumineux qui ne doivent pas être versionnés. Les fichiers `.gitkeep` permettent simplement de conserver l'arborescence vide dans Git.

## Installation

### Poetry  
Installer les dépendances du projet avec Poetry :

```bash
poetry install --no-root

# La version attendue est Python 3.12.
poetry run python --version
```
### Variables d'environnement

Créer un fichier `.env` à partir du fichier d'exemple et renseigner la clé d'API `MISTRAL_API_KEY`.

```bash
cp .env.example .env
```

### Vérification de l'environnement

Tester que les principales dépendances sont correctement installées :

```bash
# Test des dépendances
poetry run python -c "import pandas, requests, dotenv, pydantic, fastapi, langchain, faiss, mistralai; print('Imports OK')"

# Test du chargement de la configuration 
poetry run python -c "from echo_app.config import MISTRAL_MODEL, OPENAGENDA_BASE_URL; print(MISTRAL_MODEL); print(OPENAGENDA_BASE_URL)"

# Test de la présence de la clé Mistral
poetry run python -c "from echo_app.config import get_mistral_api_key; print('Mistral key OK' if get_mistral_api_key() else 'Missing key')"
```

## Commandes utiles

### Qualité du code

```bash
poetry run ruff check .
poetry run pytest
```

### Pipeline OpenAgenda

La collecte utilise une date de référence figée au `2026-05-01` afin de rendre le POC reproductible. Le pipeline transforme ensuite les événements bruts en événements nettoyés, puis en documents Markdown prêts pour le futur chunking et l’indexation FAISS.

```bash
# 1. Collecter les événements bruts
poetry run python scripts/01_fetch_openagenda_events.py

# 2. Filtrer les événements avec la date de référence du POC
poetry run python scripts/02_filter_openagenda_events.py

# 3. Nettoyer, normaliser et convertir le HTML utile en Markdown
poetry run python scripts/03_clean_openagenda_events.py

# 4. Construire les documents textuels pour le RAG
poetry run python scripts/04_build_event_documents.py
```

Fichiers générés localement :

```text
data/raw/openagenda_events_raw.json
data/processed/events_filtered.json
data/processed/events_clean.json
data/processed/events_documents.jsonl
```

Le fichier `events_documents.jsonl` contient un document par ligne, avec :

- `document_text` : texte Markdown lisible et indexable ;
- `metadata` : informations structurées conservées séparément, comme l’URL source, la ville, les dates, l’image et les coordonnées.
