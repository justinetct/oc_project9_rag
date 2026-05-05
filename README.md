# Écho - Chatbot culturel
*Projet OpenClassrooms - Concevez et déployez un système RAG.*

> Écho est le chatbot culturel développé par Puls-Events. Il permet d’interroger une base d’événements OpenAgenda à l’aide d’un système RAG combinant recherche vectorielle FAISS et génération de réponse par Mistral.

## Objectif

Concevoir un système RAG capable de répondre à des questions à partir d’événements collectés via OpenAgenda.

Les données sont récupérées depuis l’API Opendatasoft / OpenAgenda, puis nettoyées, indexées avec FAISS et utilisées par le chatbot Écho pour générer des réponses contextualisées.

## Sommaire

- [Objectif](#objectif)
- [Stack technique](#stack-technique)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Commandes utiles](#commandes-utiles)

## Stack technique

L'application s'appuiera notamment sur :

- Python 3.12
- Poetry pour la gestion de l'environnement
- LangChain pour la chaîne RAG
- FAISS pour l'index vectoriel
- Mistral AI pour le modèle de langage
- FastAPI pour l'exposition d'une API

Le rapport technique du projet sera rédigé dans :

```text
docs/rapport_technique.md
```

## Structure du projet

```text
oc_project9_rag/
├── data/                 # Données du projet
│   ├── raw/              # Données brutes collectées depuis OpenAgenda
│   ├── processed/        # Données nettoyées ou transformées
│   └── evaluation/       # Jeux de données utilisés pour évaluer le RAG
├── docs/                 # Documentation projet et notes de démonstration
├── notebooks/            # Notebooks d'exploration et d'analyse
├── scripts/              # Scripts exécutables ponctuels
│   └── fetch_openagenda_events.py  # Récupération des événements OpenAgenda bruts
├── src/                  # Code commun et fonctions utilitaires
│   ├── config.py         # Constantes, chemins et paramètres de collecte
│   ├── openagenda.py     # Client simple pour l'API OpenAgenda
│   └── utils/io.py       # Fonctions simples d'entrée / sortie
├── echo_app/             # Application principale Écho
│   ├── config.py         # Configuration spécifique à l'application
│   ├── indexing/         # Création et mise à jour de l'index vectoriel
│   ├── rag/              # Logique RAG : recherche, prompt et génération
│   └── api/main.py       # Point d'entrée FastAPI
├── tests/                # Tests automatisés
├── vector_store/         # Index FAISS généré localement
├── .env.example          # Exemple de variables d'environnement
├── pyproject.toml        # Configuration Poetry
└── README.md
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

### Collecter les événements bruts

```bash
poetry run python scripts/fetch_openagenda_events.py
```
Les événements collectés sont sauvegardés dans 'data/raw/openagenda_events_raw.json'.
