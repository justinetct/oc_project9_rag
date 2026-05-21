# Écho — Assistant RAG de recommandation culturelle

*POC d’assistant RAG pour interroger des événements culturels du Bassin d’Arcachon et du Val de l’Eyre.*

<img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
<img alt="LangChain" src="https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white">
<img alt="Mistral AI" src="https://img.shields.io/badge/Mistral_AI-FA520F?logo=mistralai&logoColor=white">
<img alt="FAISS" src="https://img.shields.io/badge/FAISS-0866FF?logo=meta&logoColor=white">
<img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white">

> Écho est l’assistant RAG développé pour Puls-Events. Il permet d’interroger une base d’événements OpenAgenda à l’aide d’un système RAG combinant recherche vectorielle FAISS et génération de réponse par Mistral.

📄 **[Rapport technique HTML](https://justinetct.github.io/oc_project9_rag/rapport_technique.html)**

## Objectif

Concevoir un système RAG capable de répondre à des questions à partir d’événements collectés via OpenAgenda.

Les données sont récupérées depuis l’API Opendatasoft / OpenAgenda, puis pré-traitées sous forme de documents Markdown indexables. Ces documents sont ensuite découpés, transformés en embeddings, indexés avec FAISS via LangChain et utilisés par Écho pour générer des réponses contextualisées.

## Sommaire

- [Stack technique](#stack-technique)
- [Documentation](#documentation)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
  - [Poetry](#poetry)
  - [Variables d'environnement](#variables-denvironnement)
  - [Vérification de l'environnement](#vérification-de-lenvironnement)
  - [Qualité du code](#qualité-du-code)
- [Pipeline OpenAgenda](#pipeline-openagenda)
- [Préparation des documents pour l'indexation](#préparation-des-documents-pour-lindexation)
- [Chaîne RAG](#chaîne-rag)
- [API FastAPI](#api-fastapi)
- [Docker](#docker)

## Stack technique

L’application s’appuie notamment sur :

- Python 3.12
- Poetry pour la gestion de l'environnement
- LangChain pour la chaîne RAG
- FAISS via LangChain pour l'index vectoriel et la recherche sémantique
- Mistral AI pour les embeddings et le modèle de langage
- FastAPI pour l'exposition d'une API
- Docker / Docker Compose pour le déploiement local reproductible de l'API

## Documentation

Les documents techniques sont regroupés dans [`docs/`](docs/) :

| Document | Contenu |
|---|---|
| [`docs/openagenda_exploration.md`](docs/openagenda_exploration.md) | Notes d'exploration initiale des données OpenAgenda. |
| [`docs/openagenda_preprocessing.md`](docs/openagenda_preprocessing.md) | Pipeline de collecte, filtrage, nettoyage et construction des documents textuels. |
| [`docs/faiss_indexing.md`](docs/faiss_indexing.md) | Stratégie de chunking, embeddings Mistral, index FAISS et recherche sémantique. |
| [`docs/rapport_technique.md`](docs/rapport_technique.md) | Rapport technique du POC : architecture, RAG, évaluation, limites et perspectives. |
| [Rapport technique HTML](https://justinetct.github.io/oc_project9_rag/rapport_technique.html) | Version HTML du rapport technique, publiée via GitHub Pages. |

## Structure du projet

```text
oc_project9_rag/
├── .dockerignore                      # Fichiers exclus du build Docker
├── .env.example                       # Exemple de variables d'environnement
├── Dockerfile                         # Image Docker de l'API FastAPI
├── docker-compose.yml                 # Lancement local de l'API via Docker Compose
├── data/                              # Données du projet
│   ├── evaluation/                    # Jeux de données utilisés pour évaluer le RAG
│   ├── processed/                     # Données nettoyées ou transformées
│   └── raw/                           # Données brutes collectées depuis OpenAgenda
├── docs/                              # Documentation du projet
├── echo_app/                          # Application principale Écho
│   ├── api/                           # API FastAPI
│   │   ├── main.py                    # Point d'entrée FastAPI et endpoints HTTP
│   │   └── schemas.py                 # Schémas Pydantic de l'API
│   ├── config.py                      # Configuration spécifique à l'application
│   ├── indexing/                      # Création et mise à jour de l'index vectoriel
│   │   ├── embeddings.py              # Génération des embeddings Mistral
│   │   ├── langchain_embeddings.py    # Adaptateur embeddings Mistral pour LangChain
│   │   ├── langchain_faiss_store.py   # Vector store FAISS LangChain
│   │   ├── rebuild.py                 # Reconstruction de l'index réutilisée par CLI et API
│   │   ├── search.py                  # Recherche sémantique via FAISS LangChain
│   │   └── faiss_store.py             # Ancien store FAISS bas niveau conservé pour les tests
│   └── rag/                           # Logique RAG : recherche, prompts et génération
│       ├── langchain_chain.py         # Assemblage des messages avec LangChain
│       ├── prompts.py                 # Prompts métier du chatbot Écho
│       └── rag_service.py             # Service RAG : recherche, contexte, prompt et génération
├── notebooks/                         # Notebooks d'exploration et d'analyse
│   ├── 01_openagenda_exploration.ipynb   # Exploration initiale des données OpenAgenda
│   ├── 02_openagenda_preprocessing.ipynb # Pré-processing des événements collectés
│   ├── 03_faiss_indexing.ipynb        # Exploration du chunking avant indexation
│   └── 04_rag_evaluation.ipynb        # Visualisation du jeu annoté et des résultats d'évaluation (scores Ragas + métriques maison)
├── pyproject.toml                     # Configuration Poetry
├── README.md
├── scripts/                           # Scripts exécutables ponctuels
│   ├── 01_fetch_openagenda_events.py  # Récupération des événements OpenAgenda bruts
│   ├── 02_filter_openagenda_events.py # Filtrage temporel des événements
│   ├── 03_clean_openagenda_events.py  # Nettoyage et normalisation des événements
│   ├── 04_build_event_documents.py    # Création des documents textuels pour le RAG
│   ├── 05_test_mistral_embeddings.py  # Test manuel des embeddings Mistral
│   ├── 06_test_semantic_search.py     # Test manuel de la recherche sémantique
│   ├── 07_test_rag_service.py         # Test manuel de la chaîne RAG complète
│   ├── 08_evaluate_rag.py             # Évaluation automatique du RAG sur le jeu annoté
│   ├── api_test.py                    # Test fonctionnel manuel de l'API FastAPI locale
│   └── rebuild_index.py               # Commande CLI de reconstruction du vector store FAISS
├── src/                               # Code commun et fonctions utilitaires
│   ├── chunking.py                    # Découpage des documents en chunks indexables
│   ├── config.py                      # Constantes, chemins et paramètres de collecte
│   ├── documents.py                   # Construction des documents textuels indexables
│   ├── openagenda.py                  # Client simple pour l'API OpenAgenda
│   ├── preprocessing.py               # Filtrage, nettoyage et normalisation des événements
│   └── utils/io.py                    # Fonctions simples d'entrée / sortie
├── tests/                             # Tests automatisés
└── vector_store/                      # Index FAISS généré localement
```

Le dépôt versionne les fichiers nécessaires au lancement local du POC, notamment `data/processed/events_documents.jsonl`. Les autres fichiers générés ou volumineux de `data/` et `vector_store/` ne sont pas versionnés ; les scripts permettent de les régénérer si besoin.

## Installation

### Poetry

Installer les dépendances du projet avec Poetry :

```bash
poetry install --no-root

# La version attendue est Python 3.12.
poetry run python --version
```

### Variables d'environnement

Créer un fichier `.env` à partir du fichier d'exemple et renseigner la clé d'API `MISTRAL_API_KEY`. Deux modèles Mistral sont distingués : `MISTRAL_MODEL` pour la génération de réponse et `MISTRAL_EMBEDDING_MODEL` pour les embeddings.

```bash
cp .env.example .env
```

### Vérification de l'environnement

Tester que les principales dépendances sont correctement installées :

```bash
# Test des dépendances
poetry run python -c "import pandas, requests, dotenv, pydantic, fastapi, langchain, faiss, mistralai; print('Imports OK')"

# Test du chargement de la configuration
poetry run python -c "from echo_app.config import MISTRAL_MODEL; from echo_app.indexing.embeddings import DEFAULT_EMBEDDING_MODEL; from src.config import OPENAGENDA_BASE_URL; print(MISTRAL_MODEL); print(DEFAULT_EMBEDDING_MODEL); print(OPENAGENDA_BASE_URL)"

# Test de la présence de la clé Mistral
poetry run python -c "from echo_app.config import get_mistral_api_key; print('Mistral key OK' if get_mistral_api_key() else 'Missing key')"
```

### Qualité du code

```bash
poetry run ruff check .
poetry run pytest -v
poetry run python -m compileall src echo_app scripts
```

Le coverage peut être recalculé avec :

```bash
poetry run pytest --cov=echo_app --cov-report=term-missing
```

Résultat :

```bash
=============================================================================================================== tests coverage ===============================================================================================================
_____________________________________________________________________________________________ coverage: platform darwin, python 3.12.13-final-0 ______________________________________________________________________________________________

Name                                         Stmts   Miss  Cover   Missing
--------------------------------------------------------------------------
echo_app/__init__.py                             0      0   100%
echo_app/api/__init__.py                         0      0   100%
echo_app/api/main.py                            76      4    95%   71-73, 163
echo_app/api/schemas.py                         42      0   100%
echo_app/config.py                               7      1    86%   19
echo_app/indexing/__init__.py                    6      0   100%
echo_app/indexing/embeddings.py                 62      5    92%   53, 70, 82, 93, 105
echo_app/indexing/faiss_store.py                63      8    87%   24, 34, 37, 41, 96, 98, 119, 121
echo_app/indexing/langchain_embeddings.py        8      0   100%
echo_app/indexing/langchain_faiss_store.py      47      3    94%   36, 92, 98
echo_app/indexing/rebuild.py                    60     28    53%   42, 68-80, 95, 98-99, 109-153
echo_app/indexing/search.py                     17      0   100%
echo_app/rag/__init__.py                         4      0   100%
echo_app/rag/langchain_chain.py                 12      0   100%
echo_app/rag/prompts.py                          4      0   100%
echo_app/rag/rag_service.py                    114      5    96%   75, 194-196, 208
echo_app/rag/ragas_compat.py                     7      5    29%   26-33
--------------------------------------------------------------------------
TOTAL                                          529     59    89%
============================================================================================================ 150 passed in 1.51s =============================================================================================================
```

Structure des tests automatisés :

```text
tests/
├── test_api.py                   # Endpoints FastAPI, validation et erreurs sans appel réseau
├── test_chunking.py              # Découpage des documents en chunks
├── test_documents.py             # Construction des documents textuels RAG
├── test_embeddings.py            # Embeddings Mistral avec appels mockés
├── test_evaluation_dataset.py    # Structure du jeu annoté d'évaluation
├── test_faiss_store.py           # Ancien store FAISS bas niveau, conservé pour compatibilité
├── test_imports.py               # Imports principaux du projet
├── test_io.py                    # Fonctions simples d'entrée / sortie
├── test_langchain_chain.py       # Construction des messages avec ChatPromptTemplate
├── test_langchain_embeddings.py  # Adaptateur Mistral compatible LangChain
├── test_langchain_faiss_store.py # Vector store FAISS LangChain : build, save, load, retriever
├── test_preprocessing.py         # Filtrage, nettoyage et normalisation des événements
├── test_rag_evaluation.py        # Fonctions de scoring, agrégation et intégration Ragas de l'évaluation RAG
├── test_rag_prompts.py           # Prompts métier de l’assistant Écho
├── test_rag_service.py           # Service RAG avec retrieval et génération mockés
├── test_rebuild_index.py         # Reconstruction du vector store FAISS LangChain
└── test_search.py                # Recherche sémantique via FAISS LangChain
```

Les tests couvrent :

- les imports principaux du projet ;
- les fonctions d’entrée / sortie ;
- le nettoyage et le pré-processing OpenAgenda ;
- la construction des documents textuels RAG ;
- le chunking des documents avant indexation ;
- la génération d'embeddings Mistral avec des tests mockés, sans appel réseau ;
- l'adaptateur d'embeddings LangChain et le vector store FAISS LangChain ;
- les prompts métier de l’assistant Écho ;
- l'assemblage des messages LangChain sans appel réseau ;
- la chaîne RAG avec recherche, filtrage des sources par écart de distance L2, construction du contexte, génération mockée et retry sur erreur de capacité Mistral (HTTP 429) ;
- la structure du jeu de test annoté d'évaluation (`data/evaluation/qa_annotated.csv`) ;
- les fonctions de calcul de l'évaluation RAG : scoring maison, préparation du dataset Ragas, fusion et agrégation des scores Ragas (mocks pandas, sans appel réseau).
- l'API FastAPI (`/health`, `/ask`, `/rebuild`, validation et erreurs) sans appel réseau ;

## Pipeline OpenAgenda

La collecte utilise une date de référence figée au `2026-05-01` afin de rendre le POC reproductible. Le pipeline transforme ensuite les événements bruts en événements nettoyés, puis en documents Markdown prêts pour le chunking et l’indexation FAISS.

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

Le fichier `data/processed/events_documents.jsonl` est versionné pour permettre la reconstruction de l’index sur un nouvel environnement sans relancer toute la collecte OpenAgenda. Pour reconstruire entièrement les données depuis zéro, lancer les scripts `01` à `04`, puis `scripts/rebuild_index.py`.

## Préparation des documents pour l'indexation

Les documents Markdown générés à partir des événements OpenAgenda sont préparés avant leur indexation vectorielle.

Le notebook `notebooks/03_faiss_indexing.ipynb` compare plusieurs approches : sans chunking, chunking par taille, chunking Markdown et chunking spaCy par phrases.

> **Modèle spaCy optionnel** :
> Cette exploration a été réalisée ponctuellement dans le notebook avec spaCy et le modèle `fr_core_news_sm`.
> spaCy n’est pas requis pour exécuter le pipeline principal de chunking, qui utilise `src/chunking.py`.
> La dépendance n'est donc pas installée par défaut dans le projet.

La stratégie de chunking retenue est volontairement simple :

- les événements courts restent en un seul chunk ;
- les événements longs sont découpés par taille avec un léger overlap ;
- chaque chunk conserve `event_id`, `chunk_id`, `chunk_index`, `chunk_count` et les métadonnées de l'événement.

Chaque chunk peut ensuite être transformé en vecteur avec le modèle d'embeddings Mistral. Ces embeddings sont utilisés pour construire le vector store FAISS LangChain. Un test manuel est disponible pour vérifier l'appel à l'API avec la clé locale :

```bash
poetry run python scripts/05_test_mistral_embeddings.py
```

Le vector store FAISS LangChain (`langchain_community.vectorstores.FAISS`) peut ensuite être reconstruit avec :

```bash
poetry run python scripts/rebuild_index.py
```

Cette commande sauvegarde l'index sous forme de `vector_store/index.faiss` + `vector_store/index.pkl` (format `save_local()` de LangChain).

Une fois l'index reconstruit, un script permet de tester la recherche sémantique sur quelques requêtes prédéfinies :

```bash
poetry run python scripts/06_test_semantic_search.py
```

## Chaîne RAG

La chaîne RAG d'Écho s'appuie sur quatre étapes simples :

```text
question utilisateur
→ retriever LangChain (.as_retriever(search_kwargs={"k": 5})) sur le vector store FAISS
→ construction du contexte et des messages LangChain (ChatPromptTemplate)
→ génération de réponse avec Mistral
```

La logique est organisée dans `echo_app/rag/` et `echo_app/indexing/` :

- [`rag_service.py`](echo_app/rag/rag_service.py) orchestre le retriever LangChain, le contexte, l'appel Mistral et les sources ;
- [`prompts.py`](echo_app/rag/prompts.py) centralise les prompts métier ;
- [`langchain_chain.py`](echo_app/rag/langchain_chain.py) construit les messages envoyés au modèle ;
- [`langchain_faiss_store.py`](echo_app/indexing/langchain_faiss_store.py) construit/charge le vector store FAISS LangChain ;
- [`langchain_embeddings.py`](echo_app/indexing/langchain_embeddings.py) expose les embeddings Mistral à LangChain.

Commandes utiles :

```bash
# Reconstruire l'index FAISS
poetry run python scripts/rebuild_index.py

# Tester manuellement la chaîne RAG complète
poetry run python scripts/07_test_rag_service.py

# Évaluer le RAG sur le jeu annoté (évaluation Ragas + métriques maison)
poetry run python scripts/08_evaluate_rag.py
```

### Évaluation Ragas

L'évaluation automatique du RAG repose désormais sur **[Ragas](https://docs.ragas.io/)** comme évaluation principale, en complément des métriques maison historiques :

- **Métriques Ragas** (évaluation principale, LLM juge Mistral + `mistral-embed`) :
  - `faithfulness` — fidélité de la réponse aux contextes utilisés ;
  - `answer_relevancy` — pertinence de la réponse vis-à-vis de la question ;
  - `context_precision` — qualité du retrieval (chunks pertinents en tête) ;
  - `context_recall` — couverture du retrieval vis-à-vis de la réponse attendue.
- **Métriques maison** (colonnes complémentaires, sans LLM juge) :
  - `keyword_match_rate`, `event_recall`, `sources_count`, `status` (`ok` / `partial` / `ko`).

Le script `scripts/08_evaluate_rag.py` rejoue les 15 questions du jeu annoté en deux étapes Mistral :

1. **15 appels** au RAG pour générer les réponses (étape 1).
2. **15 × 4 = 60 jobs Ragas** (étape 2), chaque job pouvant déclencher 1 ou 2 appels Mistral supplémentaires (LLM juge + embeddings) selon la métrique.

Les sorties sont écrites dans `data/evaluation/rag_evaluation_results.csv` (détail) et `data/evaluation/rag_evaluation_summary.json` (résumé). La sortie console de la dernière exécution validée est conservée dans [`data/evaluation/rag_evaluation.log`](data/evaluation/rag_evaluation.log) pour pouvoir relire les logs étape par étape sans relancer les ~34 minutes d'évaluation. Le script n'est pas exécuté par pytest ni en CI : les tests unitaires mockent les fonctions d'évaluation.

Le pipeline est volontairement minimal : `datasets.Dataset.from_dict` avec les colonnes `question`, `answer`, `contexts`, `ground_truth` + `ChatMistralAI` / `MistralAIEmbeddings` (langchain-mistralai) passés directement à `ragas.evaluate`. Le LLM juge Ragas utilise `mistral-large-latest` (plus fiable sur les schémas Pydantic des prompts legacy) alors que la chaîne RAG d'Écho continue d'utiliser `mistral-small-latest` côté production.

> **Note technique** : la métrique `answer_relevancy` est instanciée avec `strictness=1` (au lieu de la valeur par défaut 3) pour contourner un bug d'agrégation entre `langchain-mistralai 1.1.4` et `n > 1` complétions. Une seule question alternative est donc générée par réponse pour le calcul de similarité — score légèrement moins robuste mais reproductible. Détail dans `docs/rapport_technique.md`.

Le jeu de test annoté et les derniers résultats d'évaluation sont versionnés dans [`data/evaluation/`](data/evaluation/). Le notebook [`notebooks/04_rag_evaluation.ipynb`](notebooks/04_rag_evaluation.ipynb) permet de visualiser le jeu annoté, les résultats générés et les métriques (Ragas + maison) sans relancer les appels Mistral.

## API FastAPI

Une API FastAPI expose le système RAG via HTTP. 

Elle sépare la couche API de la logique métier : `/ask` appelle `RagService.ask()`, tandis que `/rebuild` réutilise la même fonction de reconstruction que le script CLI.


### Lancement local

```bash
poetry run uvicorn echo_app.api.main:app --reload
```

Swagger est disponible sur : <http://127.0.0.1:8000/docs>.


### Endpoints

- `GET /health` → état minimal de l'API (rapide, aucune I/O)
- `GET /metadata` → informations techniques sur le service RAG et le vector store (sans secret)
- `POST /ask` → question/réponse RAG avec sources
- `POST /rebuild` → reconstruction locale de l'index avec confirmation

Exemples :

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metadata

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels événements autour de l astronomie sont proposés ?"}'

curl -X POST http://127.0.0.1:8000/rebuild \
  -H "Content-Type: application/json" \
  -d '{"confirm": true}'
```


### Test fonctionnel manuel

Un script permet de vérifier une API déjà lancée localement :

```bash
poetry run python scripts/api_test.py
```

Par défaut, le script vérifie `/health`, `/metadata`, `/ask` et le refus de `/rebuild` sans confirmation. 

Le rebuild réel est optionnel, car il peut déclencher des appels Mistral :

```bash
poetry run python scripts/api_test.py --with-rebuild
```

## Docker

L'API FastAPI peut être lancée dans un environnement reproductible via Docker. L'image utilise Python 3.12 slim, installe les dépendances avec Poetry et démarre Uvicorn sur le port 8000.

### Prérequis

- Docker et Docker Compose installés localement.
- Fichier `.env` présent à la racine avec la clé `MISTRAL_API_KEY` (voir [Variables d'environnement](#variables-denvironnement)). Les secrets ne sont jamais embarqués dans l'image.
- Index FAISS déjà construit localement dans `vector_store/` :

```bash
poetry run python scripts/rebuild_index.py
```

### Build de l'image

```bash
docker build -t oc-project9-rag-api .
```

### Lancement avec Docker

```bash
docker run --rm --env-file .env -p 8000:8000 oc-project9-rag-api
```

Le conteneur charge l'index FAISS embarqué au premier `POST /ask` : il n'est pas reconstruit au démarrage.

### Lancement avec Docker Compose

Docker Compose simplifie le lancement local et monte les dossiers `vector_store/` et `data/processed/` depuis le dépôt, ce qui permet de rebuilder l'index localement sans reconstruire l'image :

```bash
docker compose up --build
```

Pour arrêter le service :

```bash
docker compose down
```

### Tests rapides

Une fois l'API démarrée, vérifier que tout répond :

```bash
curl http://127.0.0.1:8000/health
```

Et ouvrir la documentation Swagger dans un navigateur :

<http://127.0.0.1:8000/docs>

### Note sur /rebuild

L'endpoint `POST /rebuild` est utile en POC local pour reconstruire l'index FAISS depuis le conteneur (il déclenche des appels Mistral). En production, il devrait être protégé (authentification, rôle dédié) ou remplacé par une tâche planifiée hors du chemin HTTP public.
