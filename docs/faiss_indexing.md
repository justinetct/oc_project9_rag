# Étape 3 - Chunking et indexation FAISS

## Objectif de l'étape 3

L'étape 3 prépare les documents OpenAgenda pour la recherche vectorielle.

Le travail se fait en deux temps :

1. découper les documents si nécessaire avec une stratégie de chunking simple ;
2. transformer ensuite ces chunks en embeddings pour construire l'index FAISS.

## Rôle du chunking

Le chunking évite d'envoyer des documents trop longs tels quels dans l'index vectoriel.

L'idée n'est pas de découper systématiquement tous les événements. Un événement court reste compréhensible et cohérent en un seul bloc. En revanche, un événement long peut contenir plusieurs idées, sections ou informations pratiques. Le découper permet de récupérer plus facilement le passage pertinent au moment de la recherche.

## Stratégies comparées

Plusieurs pistes ont été comparées dans le notebook d'exploration :

- **sans chunking** : un document OpenAgenda correspond à un seul chunk ;
- **chunking par taille avec overlap** : découpe simple en morceaux fixes avec léger recouvrement ;
- **chunking Markdown** : découpe exploratoire basée sur les titres `#` et `##` déjà présents dans les documents ;
- **spaCy phrases** : test réel avec `fr_core_news_sm` pour découper les documents en phrases puis regrouper ces phrases en chunks proches de `CHUNK_SIZE`.

## Exemple comparatif réel sur l'événement `13708573`

Pour rendre la comparaison concrète, le notebook utilise notamment l'événement `13708573`, intitulé **« Initiation à l'astronomie à Lanton »**.

Les exemples ci-dessous reprennent les sorties réelles du notebook. Ils permettent de voir directement ce que chaque stratégie produit sur le même événement.

### 1. Sans chunking

Dans cette stratégie, tout l'événement reste dans un seul bloc indexable.

- 13708573_0 (1814 caractères)
```text
# Initiation à l'astronomie à Lanton

## Description
Initiation à l'astronomie et observation aux instruments.

✨ Découvrez le ciel nocturne comme vous ne l’avez jamais observé 🌙🔭

Lors de cette soirée d’observation du ciel, laissez-vous émerveiller par les beautés cachées de l’espace, visibles uniquement grâce aux télescopes et aux jumelles astronomiques.
Planètes, galaxies, nébuleuses… vous découvrirez les plus beaux objets célestes du moment.

Tout au long de la soirée, vous comprendrez :

- le parcours de vie d’une étoile, de sa naissance à sa fin
- les grands principes de la mécanique céleste et le déplacement apparent des astres

Vous apprendrez également à :

- reconnaître plusieurs constellations, et peut-être celle de votre signe zodiacal si elle est visible
- découvrir les mythes et légendes associés aux constellations

Des expériences simples et ludiques viendront accompagner les explications pour mieux comprendre le fonctionnement de notre ciel.

Ce sera aussi l’occasion de :

- découvrir comment fonctionne un télescope
- poser toutes vos questions si vous envisagez d’en acquérir un, afin de faire un choix éclairé

Dans une ambiance conviviale et un cadre naturel inhabituel, cette soirée est une invitation à lever les yeux vers les étoiles, à se détendre et à partager un moment hors du temps.

Durée : 2h30

Activité organisée par Arnaud - Animateur scientifique

Infos et réservation sur EcoNature

Conditions : Tarifs : Enfant -18 ans 10€ ; Adulte 18 ans et + 20€

## Informations pratiques
- Lieu : Blagon, 33138 Lanton, France
- Ville : Lanton
- Adresse : Blagon, 33138 Lanton, France
- Dates : du 2026-01-05 19:30 au 2027-01-03 21:30
- Mots-clés : Nature, Sortie nature, astronomie
- Mode de participation : Sur place
- Statut : Programmé

## Source
- Organisateur : EcoNature
```

Avantage : le contexte complet de l'événement est conservé.  
Limite : si le texte est très long, la recherche vectorielle peut manquer de précision.

### 2. Chunking par taille avec overlap

Dans cette stratégie, l'événement est découpé uniquement s'il dépasse `CHUNK_SIZE`. Le titre est rappelé dans chaque chunk afin que chaque morceau reste compréhensible seul.

- 13708573_0 (1143 caractères)
```text
# Initiation à l'astronomie à Lanton

## Description
Initiation à l'astronomie et observation aux instruments.

✨ Découvrez le ciel nocturne comme vous ne l’avez jamais observé 🌙🔭

Lors de cette soirée d’observation du ciel, laissez-vous émerveiller par les beautés cachées de l’espace, visibles uniquement grâce aux télescopes et aux jumelles astronomiques.
Planètes, galaxies, nébuleuses… vous découvrirez les plus beaux objets célestes du moment.

Tout au long de la soirée, vous comprendrez :

- le parcours de vie d’une étoile, de sa naissance à sa fin
- les grands principes de la mécanique céleste et le déplacement apparent des astres

Vous apprendrez également à :

- reconnaître plusieurs constellations, et peut-être celle de votre signe zodiacal si elle est visible
- découvrir les mythes et légendes associés aux constellations

Des expériences simples et ludiques viendront accompagner les explications pour mieux comprendre le fonctionnement de notre ciel.

Ce sera aussi l’occasion de :

- découvrir comment fonctionne un télescope
- poser toutes vos questions si vous envisagez d’en acquérir un, afin de faire un choix éclairé

```
- 13708573_1 (848 caractères)
```text
# Initiation à l'astronomie à Lanton

- découvrir comment fonctionne un télescope
- poser toutes vos questions si vous envisagez d’en acquérir un, afin de faire un choix éclairé

Dans une ambiance conviviale et un cadre naturel inhabituel, cette soirée est une invitation à lever les yeux vers les étoiles, à se détendre et à partager un moment hors du temps.

Durée : 2h30

Activité organisée par Arnaud - Animateur scientifique

Infos et réservation sur EcoNature

Conditions : Tarifs : Enfant -18 ans 10€ ; Adulte 18 ans et + 20€

## Informations pratiques
- Lieu : Blagon, 33138 Lanton, France
- Ville : Lanton
- Adresse : Blagon, 33138 Lanton, France
- Dates : du 2026-01-05 19:30 au 2027-01-03 21:30
- Mots-clés : Nature, Sortie nature, astronomie
- Mode de participation : Sur place
- Statut : Programmé

## Source
- Organisateur : EcoNature
```

Avantage : le nombre de chunks reste limité et les longs textes sont découpés progressivement.  
Limite : l'overlap crée une répétition visible entre les deux chunks, ici sur les lignes concernant le fonctionnement d'un télescope.

### 3. Chunking Markdown

Dans cette stratégie exploratoire, le document est découpé selon les titres Markdown `#` et `##`.

- 13708573_0 (1498 caractères)
```text
# Initiation à l'astronomie à Lanton

## Description
Initiation à l'astronomie et observation aux instruments.

✨ Découvrez le ciel nocturne comme vous ne l’avez jamais observé 🌙🔭

Lors de cette soirée d’observation du ciel, laissez-vous émerveiller par les beautés cachées de l’espace, visibles uniquement grâce aux télescopes et aux jumelles astronomiques.
Planètes, galaxies, nébuleuses… vous découvrirez les plus beaux objets célestes du moment.

Tout au long de la soirée, vous comprendrez :

- le parcours de vie d’une étoile, de sa naissance à sa fin
- les grands principes de la mécanique céleste et le déplacement apparent des astres

Vous apprendrez également à :

- reconnaître plusieurs constellations, et peut-être celle de votre signe zodiacal si elle est visible
- découvrir les mythes et légendes associés aux constellations

Des expériences simples et ludiques viendront accompagner les explications pour mieux comprendre le fonctionnement de notre ciel.

Ce sera aussi l’occasion de :

- découvrir comment fonctionne un télescope
- poser toutes vos questions si vous envisagez d’en acquérir un, afin de faire un choix éclairé

Dans une ambiance conviviale et un cadre naturel inhabituel, cette soirée est une invitation à lever les yeux vers les étoiles, à se détendre et à partager un moment hors du temps.

Durée : 2h30

Activité organisée par Arnaud - Animateur scientifique

Infos et réservation sur EcoNature

Conditions : Tarifs : Enfant -18 ans 10€ ; Adulte 18 ans et + 20€

```
- 13708573_1 (314 caractères)
```text
# Initiation à l'astronomie à Lanton

## Informations pratiques
- Lieu : Blagon, 33138 Lanton, France
- Ville : Lanton
- Adresse : Blagon, 33138 Lanton, France
- Dates : du 2026-01-05 19:30 au 2027-01-03 21:30
- Mots-clés : Nature, Sortie nature, astronomie
- Mode de participation : Sur place
- Statut : Programmé
```

Avantage : le découpage respecte bien la structure Markdown issue du pré-processing.  
Limite : la section `Informations pratiques` devient un chunk séparé assez court et moins riche sémantiquement que la description.

### 4. Chunking spaCy par phrases

Dans cette stratégie exploratoire, `spaCy` découpe le texte en phrases. Les phrases sont ensuite regroupées pour former des chunks proches de `CHUNK_SIZE`.

- 13708573_0 (1144 caractères)

```text
# Initiation à l'astronomie à Lanton

## Description
Initiation à l'astronomie et observation aux instruments.

✨ Découvrez le ciel nocturne comme vous ne l’avez jamais observé 🌙🔭

Lors de cette soirée d’observation du ciel, laissez-vous émerveiller par les beautés cachées de l’espace, visibles uniquement grâce aux télescopes et aux jumelles astronomiques.
Planètes, galaxies, nébuleuses…

vous découvrirez les plus beaux objets célestes du moment.

Tout au long de la soirée, vous comprendrez :

- le parcours de vie d’une étoile, de sa naissance à sa fin
- les grands principes de la mécanique céleste et le déplacement apparent des astres

Vous apprendrez également à :

- reconnaître plusieurs constellations, et peut-être celle de votre signe zodiacal si elle est visible
- découvrir les mythes et légendes associés aux constellations

Des expériences simples et ludiques viendront accompagner les explications pour mieux comprendre le fonctionnement de notre ciel.

Ce sera aussi l’occasion de :

- découvrir comment fonctionne un télescope
- poser toutes vos questions si vous envisagez d’en acquérir un, afin de faire un choix éclairé

```
- 13708573_1 (707 caractères)
```text
# Initiation à l'astronomie à Lanton

Dans une ambiance conviviale et un cadre naturel inhabituel, cette soirée est une invitation à lever les yeux vers les étoiles, à se détendre et à partager un moment hors du temps.

Durée : 2h30

Activité organisée par Arnaud - Animateur scientifique

Infos et réservation sur EcoNature

Conditions : Tarifs : Enfant -18 ans 10€ ; Adulte 18 ans et + 20€

## Informations pratiques
- Lieu : Blagon, 33138 Lanton, France
- Ville : Lanton
- Adresse : Blagon, 33138 Lanton, France
- Dates : du 2026-01-05 19:30 au 2027-01-03 21:30
- Mots-clés : Nature, Sortie nature, astronomie
- Mode de participation : Sur place
- Statut : Programmé

## Source
- Organisateur : EcoNature
```

Avantage : le découpage est plus naturel linguistiquement, car il évite de couper une phrase au milieu.  
Limite : le résultat reste proche du chunking par taille sur ce corpus, tout en ajoutant une dépendance plus lourde.

## Stratégie retenue

La stratégie retenue pour le code principal reste volontairement simple :

- les événements courts restent en un seul chunk ;
- les événements longs sont découpés par taille avec un léger overlap ;
- les chunks trop petits sont évités ;
- le titre est rappelé dans chaque chunk lorsque c'est possible ;
- les métadonnées de l'événement sont recopiées dans chaque chunk.

Cette stratégie limite le nombre de chunks tout en gardant assez de contexte pour la recherche vectorielle.

Le modèle `fr_core_news_sm` est utilisé uniquement pour reproduire l'exploration spaCy du notebook. Pour le réinstaller dans un nouvel environnement, il faut exécuter :

```bash
poetry run python -m spacy download fr_core_news_sm
```

Cette commande n'est pas nécessaire pour le pipeline principal de chunking, qui reste basé sur `src/chunking.py`.

## Paramètres choisis

Les paramètres de départ sont :

- `CHUNK_SIZE = 1200`
- `CHUNK_OVERLAP = 150`
- `MIN_CHUNK_SIZE = 200`

Ces valeurs sont suffisantes pour un premier POC :

- `1200` caractères laissent assez de contexte pour un événement culturel ;
- `150` caractères d'overlap limitent la perte d'information entre deux chunks ;
- `200` caractères évitent de garder des fragments trop pauvres.

## Format des chunks

Chaque chunk utilise une structure simple :

```json
{
  "chunk_id": "13708573_0",
  "event_id": "13708573",
  "chunk_index": 0,
  "chunk_count": 2,
  "chunk_text": "# Titre de l'événement\n\nContenu du chunk...",
  "metadata": {
    "title": "Titre de l'événement",
    "city": "Arcachon",
    "url": "https://openagenda.com/..."
  }
}
```

Champs principaux :

- `chunk_id` : identifiant stable et lisible ;
- `event_id` : identifiant de l'événement source ;
- `chunk_index` : position du chunk dans l'événement ;
- `chunk_count` : nombre total de chunks pour cet événement ;
- `chunk_text` : texte qui sera utilisé plus tard pour les embeddings ;
- `metadata` : copie des métadonnées utiles pour l'affichage et le traçage.

## Embeddings Mistral

Après le chunking, chaque chunk est transformé en vecteur avec le modèle `mistral-embed`.

- modèle utilisé : `mistral-embed`
- dimension attendue : `1024`
- rôle : produire les vecteurs qui seront ensuite stockés dans l'index FAISS

Un script manuel permet de vérifier rapidement l'appel au modèle :

```bash
poetry run python scripts/05_test_mistral_embeddings.py
```

Les tests unitaires des embeddings mockent l'API Mistral et ne font pas d'appel réseau.

## Index FAISS

Après la génération des embeddings, ils sont stockés dans un index FAISS local pour permettre une recherche vectorielle rapide.

- FAISS est utilisé pour retrouver les chunks les plus proches d'une requête transformée en embedding.
- Le choix retenu est `IndexFlatL2`, une structure simple qui compare directement les vecteurs avec une distance euclidienne.
- Le fichier `vector_store/index.faiss` contient uniquement les vecteurs indexés.
- Le fichier `vector_store/metadata.json` contient le mapping entre chaque position FAISS et les informations du chunk : `chunk_id`, `event_id`, `chunk_index`, `chunk_count`, `chunk_text` et `metadata`.
- Les métadonnées sont sauvegardées séparément, car FAISS ne stocke pas directement le texte source ni les informations utiles pour l'affichage.

La commande principale recommandée pour reconstruire le vector store local est :

```bash
poetry run python scripts/rebuild_index.py
```

Cette commande reconstruit le vector store complet à partir des documents préparés : elle charge `data/processed/events_documents.jsonl`, applique le chunking, génère les embeddings Mistral, construit l'index FAISS et sauvegarde l'index et les métadonnées dans `vector_store/`.
