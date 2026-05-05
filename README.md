# Écho - Chatbot culturel
*Projet OpenClassrooms - Concevez et déployez un système RAG.*

> Écho est le chatbot culturel développé par Puls-Events. Il permet d’interroger une base d’événements OpenAgenda à l’aide d’un système RAG combinant recherche vectorielle FAISS et génération de réponse par Mistral.


## Sommaire

- [Objectif](#objectif)
- [Données utilisées](#données-utilisées)
- [Stack technique](#stack-technique)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Variables d'environnement](#variables-denvironnement)
- [Vérification de l'environnement](#vérification-de-lenvironnement)
- [Qualité du code](#qualité-du-code)
- [Rapport technique](#rapport-technique)


## Objectif

Ce projet a pour objectif de concevoir un système RAG capable de répondre à des questions à partir de données collectées via OpenAgenda.

## Données utilisées

Les données utilisées proviennent d'OpenAgenda, via l'API Opendatasoft.

Elles correspondent à des événements culturels qui seront collectés, nettoyés puis indexés pour alimenter le système RAG.

À ce stade du projet, l’API OpenAgenda a été explorée et un premier échantillon brut peut être sauvegardé. Le README sera enrichi progressivement avec les commandes de collecte complète des données et de reconstruction de l’index FAISS.

## Stack technique

L'application s'appuiera notamment sur :

- Python 3.12
- Poetry pour la gestion de l'environnement
- LangChain pour la chaîne RAG
- FAISS pour l'index vectoriel
- Mistral AI pour le modèle de langage
- FastAPI pour l'exposition d'une API

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
├── src/                  # Code commun et fonctions utilitaires
│   ├── config.py         # constantes, chemins, variables d'env
│   ├── openagenda.py     # fonctions liées à l'API OpenAgenda   
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

Puis renseigner les variables nécessaires dans le fichier `.env`.

Variables principales :

| Variable | Rôle | Obligatoire |
|---|---|---|
| `MISTRAL_API_KEY` | Clé utilisée pour appeler le modèle Mistral | Oui, pour générer des réponses |
| `MISTRAL_MODEL` | Nom du modèle Mistral utilisé | Non |
| `OPENAGENDA_BASE_URL` | URL de base de l'API Opendatasoft / OpenAgenda | Non |
| `OPENAGENDA_DATASET_ID` | Identifiant du jeu de données OpenAgenda à interroger | Oui, pour collecter les événements |

L'API Opendatasoft peut être utilisée sans clé API pour les jeux de données publics. Une clé peut être nécessaire uniquement pour accéder à des données restreintes ou bénéficier de quotas plus élevés.

Le fichier `.env` ne doit jamais être versionné.

## Vérification de l'environnement

Tester que les principales dépendances sont correctement installées :

```bash
poetry run python -c "import pandas, requests, dotenv, pydantic, fastapi, langchain, faiss, mistralai; print('Imports OK')"
```

Tester le chargement de la configuration :

```bash
poetry run python -c "from echo_app.config import MISTRAL_MODEL, OPENAGENDA_BASE_URL; print(MISTRAL_MODEL); print(OPENAGENDA_BASE_URL)"
```

Tester la présence de la clé Mistral après avoir renseigné `.env` :

```bash
poetry run python -c "from echo_app.config import get_mistral_api_key; print('Mistral key OK' if get_mistral_api_key() else 'Missing key')"
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

## Rapport technique

Le rapport technique du projet sera rédigé dans :

```text
docs/rapport_technique.md
```