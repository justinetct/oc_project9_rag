# Rapport technique - Écho, assistant intelligent de recommandation d’événements culturels

## Sommaire

1. [Objectifs du projet](#1-objectifs-du-projet)
2. [Architecture du système](#2-architecture-du-système)
3. [Préparation et vectorisation des données](#3-préparation-et-vectorisation-des-données)
4. [Choix du modèle NLP](#4-choix-du-modèle-nlp)
5. [Construction de la base vectorielle](#5-construction-de-la-base-vectorielle)
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

À l’état actuel du projet, le pipeline de préparation des données, le chunking, les embeddings, la construction de l’index FAISS, la recherche sémantique et la chaîne RAG (recherche FAISS + prompting LangChain + génération Mistral) sont implémentés et testés. Les parties API finale et évaluation du RAG seront complétées dans les étapes suivantes.

## 2. Architecture du système

### Vue globale

L’architecture cible du système est la suivante :

```text
API OpenAgenda
→ collecte des événements
→ filtrage et nettoyage
→ documents Markdown
→ chunks
→ embeddings Mistral
→ index FAISS + métadonnées
→ recherche sémantique
→ contexte pour le modèle de langage
→ réponse utilisateur via API
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

Les vecteurs sont stockés dans un index FAISS, tandis que les informations lisibles sont conservées séparément dans un fichier de métadonnées.

### Intégration LLM

L’intégration avec le modèle de langage est portée par la classe `RagService`, dans `echo_app/rag/rag_service.py`. Cette classe orchestre :

1. la recherche FAISS via la fonction existante `search_similar_events()` ;
2. la construction d’un contexte numéroté à partir des chunks retrouvés ;
3. la construction d’un message utilisateur combinant le contexte et la question ;
4. l’appel à Mistral via `client.chat.complete()` ;
5. l’extraction d’une liste de sources lisibles, dédoublonnée par `event_id`.

Le modèle de génération utilisé est `mistral-small-latest`, configurable via la variable d’environnement `MISTRAL_MODEL`. Le client Mistral est créé uniquement au moment de générer une réponse, ce qui permet d’instancier `RagService` sans clé API en environnement de test.

### Exposition via API

L’exposition via API sera réalisée avec FastAPI.

L’API cible devra permettre notamment :

- de poser une question au système ;
- de récupérer une réponse générée ;
- de reconstruire l’index si nécessaire.

Cette partie sera complétée dans une étape ultérieure.

### Technologies utilisées

Les principales technologies utilisées sont :

- Python 3.12 ;
- Poetry pour la gestion de l’environnement ;
- OpenAgenda / Opendatasoft pour les données ;
- pandas pour l’exploration et la préparation des données ;
- Mistral AI pour les embeddings et la génération de réponse ;
- FAISS pour l’index vectoriel ;
- FastAPI pour l’API cible ;
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

LangChain est volontairement limité ici à la construction du prompt. La recherche vectorielle reste assurée par la couche `echo_app/indexing` (et donc par `search_similar_events()`), sans passer par un VectorStore LangChain. Aucun historique conversationnel n'est ajouté à ce stade : chaque question est traitée indépendamment, ce qui garde l'orchestration simple et explicable.

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

L’utilisation d’un système RAG réduit le risque de réponse inventée, mais ne le supprime pas complètement. Des tests d’évaluation seront nécessaires pour mesurer la qualité réelle des réponses.

## 5. Construction de la base vectorielle

### FAISS utilisé

L’index vectoriel est construit avec FAISS à partir des embeddings générés.

Le type d’index utilisé est :

```text
IndexFlatL2
```

Ce choix est adapté au POC car :

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

Deux fichiers principaux sont générés :

```text
vector_store/index.faiss
vector_store/metadata.json
```

Le fichier `index.faiss` contient les vecteurs numériques utilisés par FAISS pour la recherche.

Le fichier `metadata.json` contient les informations associées à chaque vecteur.

Ces fichiers sont générés localement et ne doivent pas être versionnés dans Git.

### Métadonnées associées

FAISS stocke les vecteurs, mais ne conserve pas directement les informations lisibles comme le titre, la ville, les dates ou l’URL de l’événement.

Le fichier `metadata.json` permet donc de faire le lien entre un résultat FAISS et le contenu affichable.

Chaque entrée contient notamment :

- `faiss_id` ;
- `chunk_id` ;
- `event_id` ;
- `chunk_index` ;
- `chunk_count` ;
- `chunk_text` ;
- `metadata`.

Le mapping entre `faiss_id` et `metadata.json` permet de retrouver le contenu textuel et les informations d’affichage après une recherche vectorielle.

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
4. construction de l’index FAISS ;
5. sauvegarde de `index.faiss` et `metadata.json` dans `vector_store/`.

Cette reconstruction est utile lorsque les données changent ou lorsque la stratégie de chunking est modifiée. Dans cette première version, l’index est reconstruit entièrement plutôt que mis à jour partiellement. Ce choix est plus simple et plus fiable pour un POC.

## 6. API et endpoints exposés

### Framework utilisé

L’API cible du projet est prévue avec FastAPI.

FastAPI est adapté au POC car il permet de créer rapidement une API Python typée, documentée automatiquement et facile à tester.

### Endpoints cibles

Les endpoints finaux seront complétés dans la suite du projet.

Les endpoints prévus sont :

```text
POST /ask
POST /rebuild
```

`/ask` permettra d’envoyer une question utilisateur et de récupérer une réponse du système RAG.

`/rebuild` pourra éventuellement servir à reconstruire l’index vectoriel si nécessaire.

À ce stade du projet, la recherche sémantique et la chaîne RAG sont disponibles côté code via la classe `RagService`, mais l’API HTTP n’est pas encore finalisée. L’endpoint `/ask` consommera directement `RagService.ask()` lors de l’étape suivante.

### Format cible des requêtes et réponses

Format cible pour `/ask` :

```json
{
  "question": "Quels événements autour de l’astronomie sont disponibles ?"
}
```

Format cible de réponse :

```json
{
  "answer": "...",
  "sources": [
    {
      "title": "Initiation à l'astronomie à Lanton",
      "city": "Lanton",
      "url": "https://openagenda.com/..."
    }
  ]
}
```

Ce format pourra évoluer lors de l’implémentation finale de l’API.

### Tests et gestion des erreurs

Les tests automatisés déjà en place couvrent :

- les imports principaux ;
- les fonctions d’entrée / sortie ;
- le nettoyage et le pré-processing OpenAgenda ;
- la construction des documents RAG ;
- le chunking ;
- les embeddings avec mocks ;
- la construction et le chargement du vector store FAISS ;
- la recherche sémantique avec mocks ;
- la chaîne RAG : construction du contexte numéroté, dédoublonnage des sources, court-circuit sur résultats vides et validation des arguments, le tout sans appel réseau ;
- les prompts métier : présence des règles essentielles dans le prompt système et format du prompt utilisateur (contexte avant question, libellés explicites) ;
- l'assemblage LangChain : structure du résultat de `build_langchain_messages` (deux messages, rôles `system` puis `user`, format dict compatible Mistral, contenus correctement injectés).

Les cas d’erreur déjà testés incluent notamment :

- requête vide ;
- `top_k` invalide ;
- vector store manquant ;
- embeddings vides ;
- dimension d’embedding incohérente.

## 7. Évaluation du système

### État actuel de l’évaluation

À ce stade, l’évaluation quantitative complète du RAG n’est pas encore implémentée. La chaîne de génération est disponible via `RagService`, ce qui permet maintenant de mesurer la qualité des réponses sur un jeu de questions annotées.

Une première validation technique a déjà été réalisée sur la recherche sémantique, et le script `scripts/07_test_rag_service.py` permet une validation manuelle de la chaîne complète sur quelques questions types.

### Jeu de test annoté

Un premier jeu de questions/réponses annoté a été créé et stocké dans `data/evaluation/qa_annotated.csv`. Il contient 15 questions et couvre plusieurs intentions du chatbot Écho : astronomie, exposition, nature, vélo, famille, commune, spectacle, patrimoine, santé, retraite, emploi et mobilité. Il intègre également un cas hors sujet lié à une demande de restaurant, qui sert à vérifier que le système refuse d’inventer une réponse hors du contexte fourni.

Chaque ligne du CSV contient cinq colonnes :

- `question` : la requête utilisateur ;
- `expected_answer` : la réponse attendue, rédigée comme description de ce que le système devrait produire ;
- `expected_keywords` : les mots-clés attendus dans la réponse (séparés par des points-virgules) ;
- `expected_event_ids` : les identifiants d’événements OpenAgenda attendus dans les sources (peut être vide pour les cas hors sujet) ;
- `comment` : une note interne sur l’intention de la question.

Ce jeu servira ensuite à l’évaluation qualitative ou quantitative du RAG, par exemple via un calcul de rappel des `event_ids` attendus et la vérification de la présence des keywords dans la réponse générée.

### Évaluation automatique

Le script `scripts/08_evaluate_rag.py` exécute une évaluation automatique simple sur l’ensemble du jeu annoté. Pour chaque question, il appelle `RagService.ask()` puis calcule trois métriques :

- `keyword_match_rate` : proportion de mots-clés attendus retrouvés dans la réponse générée (comparaison lowercase, sous-chaîne) ;
- `event_recall` : proportion d’`event_ids` attendus retrouvés dans les sources retournées ;
- `sources_count` : nombre de sources distinctes affichées à l’utilisateur.

Un `status` parmi `ok`, `partial` et `ko` est attribué à chaque ligne via des règles simples documentées dans le code. Le cas hors sujet (question sans `expected_event_ids`) est traité à part : il est considéré comme `ok` si le système ne renvoie aucune source ou s’il indique clairement qu’il ne peut pas répondre.

Les résultats détaillés sont écrits dans `data/evaluation/rag_evaluation_results.csv` (une ligne par question), et un résumé agrégé (compteurs `ok` / `partial` / `ko` et moyennes des deux taux) dans `data/evaluation/rag_evaluation_summary.json`. Ces fichiers générés ne sont pas versionnés.

Cette évaluation automatique ne remplace pas une analyse humaine de la qualité des réponses, mais elle fournit une base reproductible pour détecter les régressions et orienter les itérations futures sur le prompt ou la recherche. La stabilité de l'évaluation est renforcée par l'usage de la graine `SEED=42` (centralisée dans `src/config.py`) passée à Mistral via `random_seed`, ce qui limite la variabilité des réponses entre deux exécutions.

Le notebook `notebooks/04_rag_evaluation.ipynb` accompagne ce script : il sert à visualiser le jeu de test annoté, lire les fichiers de résultats générés et analyser les cas `partial` / `ko`. Il ne relance pas les appels Mistral à l'ouverture afin de rester consultable sans clé API.

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
→ embedding Mistral
→ recherche FAISS
→ récupération des chunks
→ récupération des métadonnées
```

Certaines requêtes composées donnent des résultats moins précis. Par exemple, une requête combinant un type d’événement et une commune peut favoriser la commune plutôt que le type d’événement.

Cette limite est normale pour une première recherche sémantique brute, sans filtre métier ni reranking.

### Évaluation cible

L’évaluation complète devra s’appuyer sur le jeu de test annoté.

Les métriques possibles sont :

* taux de récupération d’un événement attendu dans le top-k ;
* présence des mots-clés attendus dans la réponse ;
* qualité de la réponse générée ;
* absence d’invention ;
* satisfaction subjective sur un petit jeu de test.

Cette partie sera complétée dans une étape suivante, à partir de la chaîne RAG maintenant disponible et du fichier data/evaluation/qa_annotated.csv.

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
- l’API FastAPI n’est pas encore finalisée.

### Améliorations possibles

Les améliorations possibles sont :

- ajouter des filtres sur les métadonnées, par exemple ville, date ou gratuité ;
- ajouter une recherche hybride combinant recherche vectorielle et recherche par mots-clés ;
- ajouter un reranking des résultats avant génération ;
- améliorer le prompt de génération ;
- exploiter le jeu de test annoté dans un script d’évaluation ;
- évaluer automatiquement la qualité des réponses ;
- exposer le système via une API FastAPI complète ;
- préparer un déploiement avec Docker.

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
- `echo_app/indexing/faiss_store.py` : construction, sauvegarde et chargement du vector store ;
- `echo_app/indexing/search.py` : recherche sémantique dans FAISS ;
- `echo_app/rag/rag_service.py` : chaîne RAG (recherche, contexte, prompt, génération, sources) ;
- `echo_app/rag/prompts.py` : prompts métier d’Écho (prompt système et prompt utilisateur) ;
- `echo_app/rag/langchain_chain.py` : assemblage des messages du prompt via LangChain ;
- `scripts/rebuild_index.py` : reconstruction complète de l’index ;
- `scripts/06_test_semantic_search.py` : test manuel de la recherche sémantique ;
- `scripts/07_test_rag_service.py` : test manuel de la chaîne RAG complète ;
- `notebooks/03_faiss_indexing.ipynb` : exploration du chunking, de FAISS et de la recherche.

## 10. Annexes

### Commandes utiles

Reconstruction de l’index :

```bash
poetry run python scripts/rebuild_index.py
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

### Points à compléter dans la suite du projet

Les éléments suivants devront être complétés dans les prochaines étapes :

- script d’évaluation automatique du RAG ;
- API FastAPI finalisée ;
- endpoints documentés ;
- évaluation quantitative et qualitative ;
- recommandations finales après évaluation.