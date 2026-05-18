# Rapport technique - Écho, assistant intelligent de recommandation d’événements culturels

## Sommaire

1. [Objectifs du projet](#1-objectifs-du-projet)
2. [Architecture du système](#2-architecture-du-système)
3. [Préparation et vectorisation des données](#3-préparation-et-vectorisation-des-données)
4. [Choix du modèle NLP](#4-choix-du-modèle-nlp)
5. [Base vectorielle et recherche sémantique](#5-base-vectorielle-et-recherche-sémantique)
6. [API et endpoints exposés](#6-api-et-endpoints-exposés)
7. [Évaluation du système](#7-évaluation-du-système)
8. [Recommandations et perspectives](#8-recommandations-et-perspectives)
9. [Organisation du dépôt GitHub](#9-organisation-du-dépôt-github)
10. [Annexes](#10-annexes)

## 1. Objectifs du projet

### Contexte

Puls-Events souhaite proposer un assistant intelligent capable d’aider les utilisateurs à trouver des événements culturels pertinents autour du Bassin d’Arcachon.

Le projet consiste à concevoir un système RAG, c’est-à-dire un système qui combine :

- une base documentaire construite à partir d’événements OpenAgenda ;
- une recherche vectorielle pour retrouver les événements proches d’une question utilisateur ;
- un modèle de langage pour générer une réponse naturelle à partir des événements retrouvés.

L’application développée dans le cadre du POC s’appelle **Écho**.

### Problématique

Les événements culturels sont souvent décrits dans des formats hétérogènes : descriptions longues ou courtes, champs parfois incomplets, dates multiples, lieux différents, liens externes, images et métadonnées diverses.

Un système RAG répond à ce besoin car il permet de :

- rechercher dans une base documentaire locale ;
- retrouver les passages les plus pertinents pour une question ;
- générer une réponse contextualisée ;
- limiter les réponses inventées en s’appuyant sur les documents retrouvés.

L’objectif n’est donc pas seulement de poser une question à un modèle de langage, mais de lui fournir un contexte fiable issu des événements collectés.

### Objectif du POC

Le POC doit démontrer la faisabilité technique d’un assistant culturel basé sur un système RAG.

Les objectifs principaux sont :

- collecter des événements depuis OpenAgenda ;
- nettoyer et normaliser les données ;
- transformer les événements en documents textuels exploitables ;
- découper ces documents en chunks ;
- générer des embeddings ;
- construire un index vectoriel FAISS ;
- tester la recherche sémantique ;
- préparer l’intégration avec un modèle de langage ;
- exposer le système via une API.

### Périmètre

Le périmètre du POC est volontairement limité afin de rester simple et maîtrisable.

Le corpus utilisé est composé d’événements OpenAgenda autour du Bassin d’Arcachon. Les données sont préparées localement, puis sauvegardées dans le dépôt sous forme de fichiers générés non versionnés.

À l’état actuel du projet, le pipeline de préparation des données, le chunking, les embeddings, la construction du vector store FAISS LangChain, la recherche sémantique, la chaîne RAG (retriever FAISS LangChain + prompting LangChain + génération Mistral), l’API FastAPI et une première évaluation automatique sur un jeu de test annoté sont implémentés et testés.

## 2. Architecture du système

### Schéma d’architecture du système

Le schéma ci-dessous présente les principaux composants du système Écho. Il sépare le pipeline de préparation des données, l’indexation vectorielle, la couche RAG et l’API.

```text
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Préparation des données                                          │
│                                                                     │
│ API OpenAgenda                                                      │
│   → collecte des événements                                         │
│   → filtrage géographique et temporel                               │
│   → nettoyage et normalisation                                      │
│   → documents Markdown                                              │
│   → découpage en chunks                                             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. Indexation vectorielle                                           │
│                                                                     │
│ Chunks d’événements                                                 │
│   → embeddings Mistral avec mistral-embed                           │
│   → vector store FAISS LangChain                                    │
│   → sauvegarde locale dans vector_store/                            │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. Chaîne RAG                                                       │
│                                                                     │
│ Question utilisateur                                                │
│   → retriever LangChain via as_retriever(search_kwargs={"k": 5})    │
│   → récupération des documents pertinents                           │
│   → construction du contexte                                        │
│   → prompt LangChain avec ChatPromptTemplate                        │
│   → génération Mistral avec mistral-small-latest                    │
│   → réponse augmentée avec sources                                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. Exposition API                                                   │
│                                                                     │
│ Utilisateur final                                                   │
│   → API FastAPI                                                     │
│   → endpoints /health, /metadata, /ask, /rebuild                    │
│   → /ask délègue à RagService.ask(question)                         │
│   → réponse JSON : question, answer, sources                        │
└─────────────────────────────────────────────────────────────────────┘
```


### Données entrantes

Les données entrantes proviennent de l’API OpenAgenda via l’API Opendatasoft.

Le pipeline de collecte et de préparation est découpé en plusieurs scripts :

```text
scripts/01_fetch_openagenda_events.py
scripts/02_filter_openagenda_events.py
scripts/03_clean_openagenda_events.py
scripts/04_build_event_documents.py
```

Ces scripts produisent progressivement :

```text
data/raw/openagenda_events_raw.json
data/processed/events_filtered.json
data/processed/events_clean.json
data/processed/events_documents.jsonl
```

Le fichier `events_documents.jsonl` est la base utilisée pour construire le vector store.

### Prétraitement, embeddings et base vectorielle

Les événements nettoyés sont transformés en documents Markdown. Ces documents sont ensuite découpés en chunks, puis vectorisés avec le modèle d’embedding Mistral.

Les vecteurs et leurs métadonnées sont stockés dans un vector store FAISS LangChain (`langchain_community.vectorstores.FAISS`), qui les sauvegarde localement sous forme de deux fichiers `index.faiss` + `index.pkl`.

### Intégration LLM

L’intégration avec le modèle de langage est portée par la classe `RagService`, dans `echo_app/rag/rag_service.py`. Cette classe orchestre :

1. la recherche sémantique via le retriever LangChain obtenu avec `vectorstore.as_retriever(search_kwargs={"k": top_k})`, appelé par `retriever.invoke(question)` ;
2. la construction d’un contexte numéroté à partir des `Document` retrouvés ;
3. la construction d’un message utilisateur combinant le contexte et la question ;
4. l’appel à Mistral via `client.chat.complete()` ;
5. l’extraction d’une liste de sources lisibles, dédoublonnée par `event_id`.

Le modèle de génération utilisé est `mistral-small-latest`, configurable via la variable d’environnement `MISTRAL_MODEL`. Le client Mistral est créé uniquement au moment de générer une réponse, ce qui permet d’instancier `RagService` sans clé API en environnement de test.

### Exposition via API

L’API FastAPI est implémentée dans [`echo_app/api/`](../echo_app/api/) et expose quatre endpoints :

- `GET /health` : vérifie que l’API répond (statut minimal, sans I/O) ;
- `GET /metadata` : expose l’état du système RAG et du vector store (chunks indexés, disponibilité de l’index, noms des modèles Mistral) sans secret ;
- `POST /ask` : pose une question au système et retourne `{question, answer, sources}` ; délègue à `RagService.ask()` ;
- `POST /rebuild` : reconstruit localement l’index FAISS, sous condition de `confirm=true`.

La logique RAG reste isolée dans `echo_app/rag/rag_service.py` ; la couche API se contente d’appeler `RagService.ask()`. La reconstruction de l’index est portée par `echo_app.indexing.rebuild.rebuild_index()`, partagée entre `scripts/rebuild_index.py` (CLI) et `POST /rebuild` (API).

Performance : l’index FAISS est chargé en mémoire une seule fois (lazy au premier `/ask`) puis conservé en cache. Le cache n’est invalidé qu’après un `/rebuild` réussi.

Documentation interactive : Swagger sur `/docs`, généré automatiquement par FastAPI.

Un script fonctionnel [`scripts/api_test.py`](../scripts/api_test.py) permet de vérifier rapidement une API déjà lancée : `/health`, `/metadata`, `/ask` (question simple) et `/rebuild` (refus sans confirmation). Il n’est pas exécuté par pytest et n’appelle pas Mistral coûteusement par défaut.

`POST /rebuild` est documenté comme endpoint POC. En production réelle, il devrait être protégé (authentification, rôle dédié) ou remplacé par une tâche planifiée hors du chemin HTTP.

### Technologies utilisées

Les principales technologies utilisées sont :

- Python 3.12 ;
- Poetry pour la gestion de l’environnement ;
- OpenAgenda / Opendatasoft pour les données ;
- pandas pour l’exploration et la préparation des données ;
- Mistral AI pour les embeddings et la génération de réponse ;
- FAISS via LangChain pour l’index vectoriel et la recherche sémantique ;
- FastAPI pour l’API locale ;
- pytest pour les tests automatisés ;
- ruff pour la qualité du code.

## 3. Préparation et vectorisation des données

### Source de données

Les événements sont collectés depuis OpenAgenda. La collecte récupère des événements culturels, puis le pipeline applique un filtrage afin de conserver un périmètre cohérent avec le POC.

Une date de référence fixe est utilisée pour rendre le POC reproductible :

```text
2026-05-01
```

Ce choix permet d’obtenir des résultats stables pendant le développement et les démonstrations.

### Nettoyage des données

Les événements bruts contiennent des champs hétérogènes. Le nettoyage permet de produire des données plus régulières et plus faciles à indexer.

Les traitements réalisés incluent notamment :

- suppression ou conversion du HTML inutile ;
- conversion du contenu utile en Markdown simple ;
- normalisation des champs textuels ;
- conservation des dates ;
- conservation des informations de lieu ;
- extraction ou conservation des coordonnées lorsque disponibles ;
- suppression des liens bruts répétés dans le texte indexable.

Les URL ne sont pas intégrées dans le texte indexable. Elles sont conservées dans les métadonnées afin de pouvoir être affichées après la recherche.

### Construction des documents RAG

Chaque événement est transformé en document RAG.

Chaque document contient :

- `document_text` : texte Markdown utilisé pour l’indexation ;
- `metadata` : informations structurées conservées séparément.

Exemples de métadonnées conservées :

- titre ;
- ville ;
- dates ;
- URL source ;
- image ;
- coordonnées ;
- identifiant d’événement.

Cette séparation permet de garder un texte propre pour les embeddings, tout en conservant les informations utiles pour l’affichage des résultats.

### Chunking

Avant la vectorisation, les documents sont découpés en chunks.

L’objectif du chunking est d’éviter d’envoyer des textes trop longs au modèle d’embedding et d’améliorer la précision de la recherche. Un chunk représente un morceau de document qui peut être retrouvé indépendamment dans l’index FAISS.

La stratégie retenue est volontairement simple :

- les événements courts restent en un seul chunk ;
- les événements longs sont découpés par taille ;
- un léger overlap est ajouté entre deux chunks pour limiter la perte de contexte ;
- les chunks trop petits sont évités ;
- le titre de l’événement est rappelé dans les chunks découpés lorsque c’est possible.

Les paramètres principaux sont :

```text
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 200
```

Cette stratégie a été retenue car le corpus OpenAgenda utilisé contient majoritairement des documents courts. Un découpage plus complexe, par structure Markdown ou par phrases avec spaCy, a été exploré dans le notebook, mais il n’a pas été conservé dans le pipeline principal afin de garder une solution simple, robuste et facile à maintenir.

### Génération des embeddings

Chaque chunk est transformé en vecteur numérique avec le modèle d’embedding Mistral.

Le modèle utilisé est :

```text
mistral-embed
```

Ce modèle produit des vecteurs de dimension :

```text
1024
```

La génération des embeddings est faite par batchs. Un batch n’est pas un chunk : un chunk est une unité de contenu, alors qu’un batch est simplement un groupe technique de plusieurs chunks envoyés ensemble à l’API Mistral.

Cette logique permet d’éviter de faire un appel API séparé pour chaque chunk.

Le pipeline vérifie également que les embeddings retournés ont bien la dimension attendue avant de les envoyer à FAISS. Cela évite de construire un index incohérent.

## 4. Choix du modèle NLP

### Modèle d’embedding

Pour la vectorisation, le modèle retenu est :

```text
mistral-embed
```

Ce choix est cohérent avec le reste de la stack Mistral prévue pour le projet.

Les raisons principales sont :

- intégration simple via l’API Mistral ;
- modèle adapté à des textes en langage naturel ;
- dimension fixe de 1024 ;
- compatibilité avec une indexation FAISS locale.

### Modèle de génération

Le modèle de génération est utilisé pour produire la réponse finale à partir des chunks retrouvés par la recherche sémantique.

Le modèle par défaut est :

```text
mistral-small-latest
```

Il est configurable via la variable d’environnement `MISTRAL_MODEL`. L’appel est effectué via `client.chat.complete()` du SDK Mistral, dans la méthode `_generate_answer()` de `RagService`. Cette méthode est isolée pour permettre un mock simple dans les tests.

### Prompting

Le prompt système et le prompt utilisateur sont versionnés dans `echo_app/rag/prompts.py` (constante `RAG_SYSTEM_PROMPT` et fonction `build_user_prompt`). Cette séparation permet de les itérer indépendamment du code du service et de garder le cadrage métier facile à relire.

L'assemblage des messages `[system, user]` envoyés au modèle passe par LangChain (`ChatPromptTemplate`) dans `echo_app/rag/langchain_chain.py`. La fonction `build_langchain_messages(question, context)` injecte le prompt système et le prompt utilisateur dans le template, puis convertit les messages LangChain en simples dictionnaires `{"role", "content"}` directement compatibles avec `client.chat.complete()` du SDK Mistral.

LangChain joue désormais un rôle plus complet dans la chaîne RAG. La recherche vectorielle passe par le vector store FAISS fourni par LangChain (`langchain_community.vectorstores.FAISS`), exposé à `RagService` sous forme de retriever via `.as_retriever(search_kwargs={"k": top_k})`. La récupération des chunks pertinents se fait donc via `retriever.invoke(question)`, et la construction des messages reste pilotée par `ChatPromptTemplate`. Aucun historique conversationnel n'est ajouté à ce stade : chaque question est traitée indépendamment, ce qui garde l'orchestration simple et explicable.

Le prompt envoyé à Mistral est découpé en deux messages :

- un message **système** qui définit la persona Écho et fixe les règles :
  - répondre uniquement à partir du contexte fourni ;
  - ne jamais inventer d’événement, de date, de lieu, de prix ou d’URL ;
  - rester en français, dans un ton clair, utile et bienveillant ;
  - reconnaître explicitement les cas où le contexte ne permet pas de répondre ;
  - citer les événements utilisés par leur titre et leur ville lorsque c’est pertinent ;
  - rester concis (2 à 5 phrases dans la plupart des cas) ;
  - pour une question hors sujet, indiquer poliment que le chatbot répond uniquement sur les événements présents dans le contexte.

- un message **utilisateur** qui contient :
  - les chunks retrouvés par FAISS, numérotés `[1]` à `[N]` et séparés par une ligne `---`, chacun précédé d’une en-tête `[n] titre — ville (date)` ;
  - la question utilisateur à la suite du contexte.

Le paramètre `temperature=0.2` est utilisé pour limiter la créativité du modèle et privilégier des réponses ancrées dans le contexte.

Si la recherche FAISS retourne zéro résultat, le service court-circuite l’appel au modèle et renvoie directement une réponse explicite (« Je n’ai trouvé aucun événement pertinent pour votre question. ») avec une liste de sources vide. Cela évite un appel API inutile et garantit un comportement déterministe.

### Limites du modèle

La qualité des réponses dépendra de plusieurs facteurs :

- qualité des descriptions OpenAgenda ;
- pertinence des chunks retrouvés ;
- capacité du modèle à respecter le contexte fourni ;
- présence ou absence d’informations suffisantes dans les événements collectés.

L’utilisation d’un système RAG réduit le risque de réponse inventée, mais ne le supprime pas complètement. Une première évaluation automatique a été mise en place sur un jeu annoté de 15 questions ; elle devra être enrichie pour obtenir une mesure plus robuste.

## 5. Base vectorielle et recherche sémantique
#### Intégration avec LangChain

Le système utilise désormais le vector store FAISS fourni par LangChain (`langchain_community.vectorstores.FAISS`). Les chunks OpenAgenda sont convertis en objets `Document`, indexés avec les embeddings Mistral, puis sauvegardés localement dans `vector_store/`.

Au moment de répondre à une question, `RagService` recharge cet index, crée un retriever avec `.as_retriever(search_kwargs={"k": 5})`, récupère les documents pertinents, puis injecte leur contenu dans un `ChatPromptTemplate` avant l’appel à Mistral.

Cette approche garde le pipeline métier existant tout en utilisant les abstractions LangChain standard : vector store, retriever et prompt template.

### Base vectorielle et recherche sémantique

Le vector store LangChain s'appuie en interne sur un index `IndexFlatL2` construit par `langchain_community.vectorstores.FAISS`.

Ce type d'index est adapté au POC car :

- le volume de données est faible ;
- l’index est simple à construire ;
- la recherche est exacte ;
- le fonctionnement est facile à expliquer.

`IndexFlatL2` utilise une distance L2 pour comparer les vecteurs. Plus la distance est faible, plus le chunk est considéré comme proche de la requête utilisateur.

Dans le corpus actuel, le pipeline produit :

```text
138 documents OpenAgenda
→ 155 chunks
→ 155 embeddings
→ 155 vecteurs dans FAISS
```

Le nombre de vecteurs est supérieur au nombre de documents car certains documents longs sont découpés en plusieurs chunks.

### Stratégie de persistance

Le vector store est sauvegardé localement dans le dossier :

```text
vector_store/
```

Le format actif est celui produit par `vectorstore.save_local()` de LangChain, qui génère deux fichiers :

```text
vector_store/index.faiss
vector_store/index.pkl
```

Le fichier `index.faiss` contient les vecteurs numériques utilisés par FAISS pour la recherche.

Le fichier `index.pkl` contient le docstore LangChain : il associe chaque vecteur à un `Document` (texte du chunk + métadonnées de l'événement).

Ces fichiers sont générés localement et ne doivent pas être versionnés dans Git. Le script de reconstruction supprime explicitement un éventuel ancien `metadata.json` résiduel pour éviter toute ambiguïté avec l'ancien format maison.

### Métadonnées associées

Chaque vecteur est associé à un `Document` LangChain. Les métadonnées sont stockées dans `document.metadata` (dict à plat) et contiennent notamment :

- `event_id` ;
- `chunk_id` ;
- `chunk_index` ;
- `chunk_count` ;
- `title`, `city`, `start_date`, `url` (issus des métadonnées événement) ;
- les autres champs OpenAgenda utiles (`location_name`, `latitude`, `longitude`, etc.).

LangChain s'occupe du mapping entre l'index FAISS et le `Document` : après une recherche vectorielle, `retriever.invoke(question)` retourne directement une liste de `Document`, avec leur `page_content` (texte du chunk) et leur `metadata` prêts à l'emploi.

### Reconstruction de l’index

L’index peut être reconstruit avec la commande suivante :

```bash
poetry run python scripts/rebuild_index.py
```

Cette commande régénère entièrement le vector store à partir des documents pré-processés.

Elle exécute les étapes suivantes :

1. chargement des documents depuis `data/processed/events_documents.jsonl` ;
2. construction des chunks ;
3. génération des embeddings avec Mistral ;
4. conversion des chunks en `Document` LangChain ;
5. construction du vector store FAISS LangChain (`FAISS.from_embeddings`) ;
6. sauvegarde de `index.faiss` et `index.pkl` dans `vector_store/` via `save_local()`.

Cette reconstruction est utile lorsque les données changent ou lorsque la stratégie de chunking est modifiée. Dans cette première version, l’index est reconstruit entièrement plutôt que mis à jour partiellement. Ce choix est plus simple et plus fiable pour un POC.

## 6. API et endpoints exposés

### Framework utilisé

L’API du projet est implémentée avec FastAPI, dans le dossier [`echo_app/api/`](../echo_app/api/).

FastAPI est adapté au POC car il permet de créer rapidement une API Python typée, documentée automatiquement via Swagger et facile à tester avec `TestClient`.

### Endpoints exposés

L’API expose quatre endpoints principaux :

| Endpoint | Rôle |
|---|---|
| `GET /health` | Vérifie que l’API répond avec un statut minimal. |
| `GET /metadata` | Expose des informations techniques non sensibles sur le système RAG : disponibilité du vector store, nombre de chunks, modèles utilisés, dernier rebuild. |
| `POST /ask` | Reçoit une question utilisateur et retourne une réponse générée avec ses sources. |
| `POST /rebuild` | Reconstruit localement l’index FAISS avec une confirmation explicite (`confirm=true`). |

La documentation interactive Swagger est générée automatiquement par FastAPI et disponible sur `/docs` lorsque l’API est lancée localement.

### Séparation API / logique métier

La couche API ne réimplémente pas la logique RAG. L’endpoint `POST /ask` appelle directement `RagService.ask(question)`, qui orchestre le retriever FAISS LangChain, la construction du contexte, le prompt LangChain et l’appel Mistral.

La reconstruction de l’index est portée par `echo_app.indexing.rebuild.rebuild_index()`. Cette fonction est utilisée à la fois par le script CLI `scripts/rebuild_index.py` et par l’endpoint `POST /rebuild`, ce qui évite de dupliquer la logique.

### Format des requêtes et réponses

Exemple de requête pour `/ask` :

```json
{
  "question": "Quels événements autour de l’astronomie sont disponibles ?"
}
```

Exemple de réponse :

```json
{
  "question": "Quels événements autour de l’astronomie sont disponibles ?",
  "answer": "...",
  "sources": [
    {
      "event_id": "13708573",
      "title": "Initiation à l'astronomie à Lanton",
      "city": "Lanton",
      "start_date": "2026-01-05T19:30:00+00:00",
      "url": "https://openagenda.com/..."
    }
  ]
}
```

`GET /metadata` n’expose pas de clé API ni de chemin local absolu. Les noms de modèles (`mistral-embed`, `mistral-small-latest`) sont des informations techniques non sensibles.

### Tests et gestion des erreurs

Les tests API sont regroupés dans `tests/test_api.py`. Ils utilisent `TestClient` et des fakes pour éviter tout appel réel à Mistral ou tout chargement réel de FAISS pendant les tests unitaires.

Les principaux comportements testés sont :

- `/health` retourne un statut minimal ;
- `/metadata` retourne les informations techniques attendues sans appel Mistral ni chargement FAISS ;
- `/ask` délègue bien à `RagService.ask()` ;
- une question vide ou composée uniquement d’espaces retourne une erreur claire ;
- `/rebuild` refuse une reconstruction sans `confirm=true` ;
- une erreur inattendue pendant `/ask` ou `/rebuild` retourne une erreur générique sans exposer de trace Python.

Un script fonctionnel manuel [`scripts/api_test.py`](../scripts/api_test.py) permet également de tester une API déjà lancée localement. Par défaut, il vérifie `/health`, `/metadata`, `/ask` et le refus de `/rebuild` sans confirmation. Le rebuild réel est optionnel afin d’éviter des appels Mistral coûteux pendant un test rapide.

### Limites de l’API

`POST /rebuild` est un endpoint utile pour le POC local, mais il est sensible : il peut reconstruire l’index, déclencher des appels embeddings et invalider le cache du retriever. En production, il devrait être protégé par une authentification, limité à un rôle administrateur ou remplacé par une tâche interne planifiée hors du chemin HTTP public.

## 7. Évaluation du système

### État actuel de l’évaluation

L’évaluation automatique est implémentée via le script `scripts/08_evaluate_rag.py`. Elle rejoue les 15 questions du jeu annoté sur `RagService.ask(question, include_contexts=True)` et calcule deux familles de métriques :

- **Évaluation principale — Ragas** (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`), calculée par un LLM juge Mistral (`mistral-small-latest` + `mistral-embed`) ; elle mesure la fidélité de la réponse au contexte récupéré et la qualité du retrieval ;
- **Métriques maison complémentaires** (`keyword_match_rate`, `event_recall`, `sources_count`, `status`) conservées comme signaux d’analyse lisibles à la main, sans dépendre du juge.

Cette évaluation donne une base reproductible pour suivre la qualité des réponses, tout en restant volontairement limitée : elle ne remplace pas une analyse humaine et les scores Ragas peuvent légèrement varier d’une exécution à l’autre malgré `temperature=0` côté juge.

Une validation manuelle complémentaire reste possible via `scripts/07_test_rag_service.py` (chaîne RAG complète sur quelques questions types) et `scripts/06_test_semantic_search.py` (recherche sémantique seule).

### Jeu de test annoté

Un premier jeu de questions/réponses annoté a été créé et stocké dans [`data/evaluation/qa_annotated.csv`](../data/evaluation/qa_annotated.csv). Il contient 15 questions couvrant plusieurs intentions du chatbot Écho : astronomie, exposition, nature, vélo, famille, commune, spectacle, patrimoine, santé, retraite, emploi et mobilité.

> [!NOTE]
> Le jeu contient aussi un cas hors sujet lié à une demande de restaurant, afin de vérifier que le système ne force pas une réponse quand le contexte ne le permet pas.

Chaque ligne du CSV contient cinq colonnes :

- `question` : la requête utilisateur ;
- `expected_answer` : la réponse attendue, rédigée comme description de ce que le système devrait produire ;
- `expected_keywords` : les mots-clés attendus dans la réponse (séparés par des points-virgules) ;
- `expected_event_ids` : les identifiants d’événements OpenAgenda attendus dans les sources (peut être vide pour les cas hors sujet) ;
- `comment` : une note interne sur l’intention de la question.

Exemples de lignes du jeu annoté :

| Question | Attendu | Type de cas |
|---|---|---|
| Quels événements autour de l'astronomie sont disponibles ? | Retrouver l’événement d’initiation à l’astronomie à Lanton. | Requête thématique précise |
| Peux-tu me conseiller un restaurant à Arcachon ? | Répondre prudemment que le contexte ne permet pas de recommander un restaurant. | Cas hors sujet |

Ce jeu sert de base à l’évaluation automatique : le script vérifie notamment les sources attendues et la présence de mots-clés dans la réponse générée.

### Évaluation automatique

Le script `scripts/08_evaluate_rag.py` exécute l’évaluation complète sur les 15 questions annotées. Le format est volontairement minimal :

- `datasets.Dataset.from_dict` avec les colonnes `question`, `answer`, `contexts`, `ground_truth` ;
- `ChatMistralAI` + `MistralAIEmbeddings` (langchain-mistralai) passés directement à `ragas.evaluate` (sans wrapper Ragas explicite, qui est par ailleurs déprécié) ;
- 4 métriques Ragas standards : `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.

Le **LLM juge** utilisé par Ragas est `mistral-large-latest`. Il est volontairement différent du modèle de production de la chaîne RAG (`mistral-small-latest`) : `mistral-small` respecte moins bien les schémas Pydantic attendus par les prompts legacy de Ragas (échec récurrent du parser `StringIO`), tandis que `mistral-large` les suit fidèlement. La chaîne RAG d'Écho côté production n'est pas affectée par ce choix, qui ne concerne que l'évaluation.

**Pipeline en deux étapes** :

1. **15 appels Mistral** (étape 1) — pour chaque question, le script appelle `RagService.ask(question, include_contexts=True)` afin de récupérer la réponse, les sources et les chunks utilisés. Un log lisible par ligne affiche `chunks=… | sources=… | attendu=… | status=…`.
2. **15 × 4 = 60 jobs Ragas** (étape 2) — chaque job peut déclencher 1 ou 2 appels Mistral supplémentaires (LLM juge + embeddings) selon la métrique.

**Métriques Ragas (évaluation principale)** :

- `faithfulness` : la réponse reste-t-elle fidèle aux contextes récupérés (pas d’invention) ?
- `answer_relevancy` : la réponse est-elle réellement pertinente vis-à-vis de la question posée ?
- `context_precision` : les chunks pertinents sont-ils bien placés en tête du retrieval ?
- `context_recall` : le retrieval couvre-t-il bien la réponse de référence attendue ?

> [!WARNING]
> **Note technique sur `answer_relevancy`** : la métrique est instanciée explicitement avec `AnswerRelevancy(strictness=1)` au lieu du singleton par défaut (`strictness=3`). Le défaut plante en effet avec `langchain-mistralai 1.1.4` (`TypeError: unsupported operand type(s) for +=: 'dict' and 'dict'` lors de l’agrégation des `n=3` complétions parallèles). Avec `strictness=1`, une seule question alternative est générée par réponse pour le calcul de similarité — score légèrement moins robuste mais reproductible.

**Métriques maison (colonnes complémentaires)** — calculées sans LLM juge, lisibles à la main :

- `keyword_match_rate` : proportion de mots-clés attendus retrouvés dans la réponse (lowercase, sous-chaîne) ;
- `event_recall` : proportion d’`event_ids` attendus retrouvés dans les sources ;
- `sources_count` : nombre de sources distinctes affichées à l’utilisateur ;
- `status` (`ok` / `partial` / `ko`) résumant la cohérence globale :

| Status | Signification |
|---|---|
| `ok` | La réponse retrouve au moins une source attendue et suffisamment de mots-clés, ou refuse correctement un cas hors sujet. |
| `partial` | La réponse est partiellement pertinente, mais une source attendue ou une partie des mots-clés manque. |
| `ko` | La réponse ne retrouve ni les sources attendues ni les informations clés. |

Les résultats sont versionnés pour conserver une trace de la dernière évaluation :

- [`data/evaluation/rag_evaluation_results.csv`](../data/evaluation/rag_evaluation_results.csv) : résultats détaillés, une ligne par question, colonnes maison + 4 colonnes Ragas ;
- [`data/evaluation/rag_evaluation_summary.json`](../data/evaluation/rag_evaluation_summary.json) : résumé agrégé (compteurs status, moyennes maison et moyennes Ragas) ;
- [`data/evaluation/rag_evaluation.log`](../data/evaluation/rag_evaluation.log) : sortie console de la dernière exécution validée (étape 1 ligne par ligne + résumé final), conservée pour pouvoir relire les logs sans relancer les ~34 minutes d'évaluation.

> [!IMPORTANT]
> Cette évaluation ne remplace pas une revue humaine, mais elle donne une base reproductible pour repérer les régressions. La génération est stabilisée avec `SEED=42` et `temperature=0.2` côté chaîne RAG, `temperature=0` côté juge Ragas.

Le notebook [`notebooks/04_rag_evaluation.ipynb`](../notebooks/04_rag_evaluation.ipynb) permet de visualiser le jeu annoté, les résultats générés et les cas `partial` / `ko`. Il lit les fichiers CSV/JSON déjà produits et ne relance pas les appels Mistral à l’ouverture.


### Analyse des erreurs

L’évaluation lancée sur les 15 questions annotées avec le pipeline final donne le résumé suivant :

<!-- metric-cards -->
```json
{
  "total": 15,
  "ok": 14,
  "partial": 1,
  "ko": 0,
  "average_keyword_match_rate": 0.83,
  "average_event_recall": 0.811,
  "average_faithfulness": 0.927,
  "average_answer_relevancy": 0.828,
  "average_context_precision": 0.942,
  "average_context_recall": 0.867
}
```

Les 4 métriques Ragas sont calculées pour les 15 questions (0 NaN), et confirment que la chaîne RAG est très fiable :

- `faithfulness` ≈ 0.927 : très peu d’invention, les réponses restent fidèles aux chunks récupérés ;
- `context_precision` ≈ 0.942 : les chunks pertinents sont systématiquement bien placés en tête du retrieval ;
- `context_recall` ≈ 0.867 : la couverture du retrieval est solide vis-à-vis des réponses attendues ;
- `answer_relevancy` ≈ 0.828 : les réponses générées répondent bien aux questions, à l’exception du cas hors sujet (voir ci-dessous).

Une seule question est classée `partial` :

| Question | Observation | Interprétation |
|---|---|---|
| Peux-tu me conseiller un restaurant à Arcachon ? | `keyword_match_rate` 0.333, `event_recall` 0 (rien attendu côté event_ids) ; côté Ragas, `faithfulness` 0.667, `context_precision` 0.806, `context_recall` 1.0, **`answer_relevancy` 0.0**. Le système répond prudemment mais 4 sources sont tout de même remontées. | Le score `answer_relevancy` à 0 est cohérent et attendu : Ragas détecte que la réponse (un refus) n’est pas alignée avec la question posée. C’est exactement ce qu’on veut sur un cas hors sujet, mais le status maison reste `partial` parce que des sources sont remontées. La détection automatique du hors sujet côté chaîne RAG reste perfectible. |

Aucun cas n’est classé `ko`.

**Limites connues de l’évaluation :**

- jeu de seulement 15 questions, à étendre pour stabiliser les moyennes Ragas ;
- l’exécution déclenche 15 appels Mistral pour la chaîne RAG puis 60 jobs Ragas (15 × 4 métriques, plusieurs appels LLM possibles par job) ; elle consomme du quota et n’est pas lancée en CI ;
- le plan Mistral Experiment (gratuit) impose ~1 RPS implicite, ce qui force `RAGAS_MAX_WORKERS=1` (sériel) pour ne pas perdre de scores en 429. Conséquence : l’évaluation complète prend ~30-35 min plutôt que ~5 min en parallèle. Avec un plan Scale (6 RPS officiels), `max_workers` peut être remonté à 4-8 ;
- les scores Ragas peuvent légèrement varier d’une exécution à l’autre malgré `temperature=0` côté juge ;
- `answer_relevancy` utilise `strictness=1` (1 question alternative générée au lieu de 3 par défaut), ce qui rend le score un peu moins stable mais évite le bug d’agrégation avec `langchain-mistralai 1.1.4` ;
- la détection automatique du hors sujet reste perfectible côté chaîne RAG (le seul cas `partial` est lié à ce point).

Les améliorations réalistes seraient :

- ajouter des filtres par commune, date ou gratuité ;
- améliorer la détection des questions hors sujet ;
- ajouter un reranking pour mieux sélectionner les événements les plus pertinents ;
- enrichir le jeu de test annoté avec davantage de cas ;
- passer à un plan Mistral payant pour paralléliser l’évaluation et relever `strictness` sur `answer_relevancy` dès qu’une version corrigée de `langchain-mistralai` rendra l’agrégation `n > 1` fiable.

### Validation technique actuelle

La recherche sémantique peut être vérifiée manuellement avec le script suivant :

```bash
poetry run python scripts/06_test_semantic_search.py
```

Exemples de requêtes utilisées :

- `astronomie` ;
- `concert à Andernos` ;
- `activité en famille` ;
- `exposition Bassin d’Arcachon` ;
- `événement gratuit`.

### Résultats observés

La requête `astronomie` retrouve bien les chunks associés à l’événement :

```text
Initiation à l'astronomie à Lanton
```

Cela valide le fonctionnement de la chaîne :

```text
requête utilisateur
→ embedding Mistral via l’adaptateur LangChain
→ retriever FAISS LangChain
→ récupération des Documents pertinents
→ contexte RAG + sources
```

Certaines requêtes composées donnent des résultats moins précis. Par exemple, une requête combinant un type d’événement et une commune peut favoriser la commune plutôt que le type d’événement.

Cette limite est normale pour une première recherche sémantique brute, sans filtre métier ni reranking.

### Évaluation cible

L’évaluation actuelle (Ragas + métriques maison) reste à étendre au-delà de ce premier jeu de 15 questions. Les pistes principales sont :

* élargir le jeu annoté pour couvrir plus largement les intentions utilisateur et stabiliser les moyennes Ragas ;
* mesurer la précision en plus du rappel côté métriques maison (pour pénaliser les réponses qui remontent trop d’événements hors sujet) ;
* intégrer une note qualitative humaine sur un sous-ensemble pour calibrer les seuils du `status` et croiser avec les scores Ragas ;
* envisager l’exécution de l’évaluation en intégration continue (avec budget LLM dédié) dès qu’une couverture suffisante sera atteinte.

## 8. Recommandations et perspectives

### Ce qui fonctionne bien

Le POC valide déjà plusieurs briques importantes :

- les données OpenAgenda peuvent être collectées et préparées ;
- les événements peuvent être transformés en documents Markdown ;
- le chunking produit des unités de recherche exploitables ;
- les embeddings Mistral sont générés correctement ;
- l’index FAISS est construit et persistant ;
- la recherche sémantique fonctionne sur des requêtes simples ;
- les métadonnées permettent de retrouver les informations affichables ;
- la chaîne RAG combine recherche, prompt contrôlé et génération Mistral pour produire une réponse structurée avec sources.

### Limites du POC

Cette première version présente plusieurs limites :

- la qualité dépend fortement des descriptions OpenAgenda ;
- les événements très courts donnent parfois peu de contexte au modèle d’embedding ;
- la recherche sémantique brute ne gère pas encore parfaitement les requêtes composées ;
- les filtres métier par date, commune ou gratuité ne sont pas encore appliqués directement ;
- l’index doit être reconstruit lorsque les données changent ;
- l’endpoint `/rebuild` est disponible pour le POC local, mais devrait être protégé ou externalisé en production.

### Améliorations possibles

Les améliorations possibles sont :

- ajouter des filtres sur les métadonnées, par exemple ville, date ou gratuité ;
- ajouter une recherche hybride combinant recherche vectorielle et recherche par mots-clés ;
- ajouter un reranking des résultats avant génération ;
- améliorer le prompt de génération ;
- enrichir l’évaluation automatique (jeu de test plus grand, métriques sémantiques) ;
- affiner la détection du hors sujet pour éviter les faux négatifs côté métrique ;
- préparer le déploiement de l’API avec Docker et sécuriser les endpoints sensibles.

## 9. Organisation du dépôt GitHub

L’organisation du dépôt est la suivante :

```text
oc_project9_rag/
├── data/                  # Données brutes et transformées, non versionnées
├── docs/                  # Documentation projet et rapport technique
├── echo_app/              # Code de l’application Écho
│   ├── api/               # API FastAPI cible
│   ├── indexing/          # Embeddings, FAISS et recherche sémantique
│   └── rag/               # Chaîne RAG 
├── notebooks/             # Notebooks d’exploration et de validation
├── scripts/               # Scripts exécutables du pipeline
├── src/                   # Fonctions communes de collecte, nettoyage, documents et chunking
├── tests/                 # Tests automatisés
└── vector_store/          # Index FAISS généré localement, non versionné
```

### Fichiers et dossiers clés

Les principaux fichiers et dossiers sont :

- `src/openagenda.py` : client simple pour l’API OpenAgenda ;
- `src/preprocessing.py` : nettoyage et normalisation des événements ;
- `src/documents.py` : construction des documents RAG ;
- `src/chunking.py` : découpage des documents en chunks ;
- `echo_app/indexing/embeddings.py` : génération des embeddings Mistral ;
- `echo_app/indexing/langchain_embeddings.py` : adaptateur embeddings Mistral compatible LangChain ;
- `echo_app/indexing/langchain_faiss_store.py` : construction, sauvegarde et chargement du vector store FAISS LangChain ;
- `echo_app/indexing/rebuild.py` : reconstruction de l’index réutilisée par le script CLI et l’API ;
- `echo_app/indexing/search.py` : recherche sémantique via le vector store FAISS LangChain ;
- `echo_app/indexing/faiss_store.py` : ancienne implémentation FAISS bas niveau conservée pour compatibilité et tests ;
- `echo_app/rag/rag_service.py` : chaîne RAG (recherche, contexte, prompt, génération, sources) ;
- `echo_app/rag/prompts.py` : prompts métier d’Écho (prompt système et prompt utilisateur) ;
- `echo_app/rag/langchain_chain.py` : assemblage des messages du prompt via LangChain ;
- `echo_app/api/main.py` : endpoints FastAPI `/health`, `/metadata`, `/ask` et `/rebuild` ;
- `echo_app/api/schemas.py` : schémas Pydantic de l’API ;
- `scripts/rebuild_index.py` : wrapper CLI de reconstruction de l’index ;
- `scripts/api_test.py` : test fonctionnel manuel de l’API locale ;
- `scripts/06_test_semantic_search.py` : test manuel de la recherche sémantique ;
- `scripts/07_test_rag_service.py` : test manuel de la chaîne RAG complète ;
- `scripts/08_evaluate_rag.py` : évaluation automatique sur le jeu de test annoté ;
- `notebooks/03_faiss_indexing.ipynb` : exploration du chunking, de FAISS et de la recherche ;
- `notebooks/04_rag_evaluation.ipynb` : visualisation du jeu annoté et des résultats d’évaluation.

## 10. Annexes

### Commandes utiles

Reconstruction de l’index :

```bash
poetry run python scripts/rebuild_index.py
```

Lancement de l’API locale :

```bash
poetry run uvicorn echo_app.api.main:app --reload
```

Test fonctionnel de l’API locale déjà lancée :

```bash
poetry run python scripts/api_test.py
```

Test manuel de la recherche sémantique :

```bash
poetry run python scripts/06_test_semantic_search.py
```

Test manuel de la chaîne RAG complète :

```bash
poetry run python scripts/07_test_rag_service.py
```

Tests automatisés :

```bash
poetry run pytest -v
```

Qualité du code :

```bash
poetry run ruff check .
```

### Exemple de recherche sémantique

Exemple de requête :

```text
astronomie
```

Résultat observé dans les premiers résultats :

```text
Initiation à l'astronomie à Lanton
```

Ce résultat montre que la recherche sémantique retrouve correctement un événement dont le contenu est proche de la requête utilisateur.
