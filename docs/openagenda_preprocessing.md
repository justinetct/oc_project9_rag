# Pré-processing OpenAgenda

## Sommaire

- [Objectif](#objectif)
- [Fichiers du pipeline](#fichiers-du-pipeline)
- [Commandes du pipeline](#commandes-du-pipeline)
- [1. Filtrage temporel](#1-filtrage-temporel)
  - [Date de référence](#date-de-référence)
  - [Règle de filtrage](#règle-de-filtrage)
- [2. Nettoyage et normalisation](#2-nettoyage-et-normalisation)
  - [Champs conservés](#champs-conservés)
  - [Nettoyage appliqué](#nettoyage-appliqué)
  - [Champs volontairement exclus](#champs-volontairement-exclus)
  - [Exemple court d'événement nettoyé](#exemple-court-dévénement-nettoyé)
- [3. Construction des documents textuels](#3-construction-des-documents-textuels)
  - [Différence entre données structurées, `document_text` et `metadata`](#différence-entre-données-structurées-document_text-et-metadata)
  - [Champs utilisés dans `document_text`](#champs-utilisés-dans-document_text)
  - [Métadonnées conservées](#métadonnées-conservées)
  - [Exemple court de `document_text` Markdown](#exemple-court-de-document_text-markdown)

## Objectif

Le pré-processing OpenAgenda prépare un corpus cohérent pour le RAG à partir des événements bruts collectés via l'API OpenAgenda.

Le pipeline suit trois étapes :

1. filtrer les événements encore actifs ou à venir ;
2. nettoyer et normaliser les champs utiles ;
3. construire des documents textuels lisibles et indexables.

## Fichiers du pipeline

- entrée brute : `data/raw/openagenda_events_raw.json`
- sortie du filtrage : `data/processed/events_filtered.json`
- sortie du nettoyage : `data/processed/events_clean.json`
- sortie documents : `data/processed/events_documents.jsonl`

## Commandes du pipeline

```bash
poetry run python scripts/01_fetch_openagenda_events.py
poetry run python scripts/02_filter_openagenda_events.py
poetry run python scripts/03_clean_openagenda_events.py
poetry run python scripts/04_build_event_documents.py
```

## 1. Filtrage temporel

### Date de référence

Le POC utilise une date de référence figée :

```text
2026-05-01
```

### Règle de filtrage

Le filtre principal conserve les événements dont la dernière date de fin est supérieure ou égale à cette date :

```text
lastdate_end >= 2026-05-01
```

Si `lastdate_end` est absent, le fallback est :

1. `firstdate_end`
2. `firstdate_begin`

Ce choix évite d'indexer des événements déjà terminés et rend le POC reproductible pour les tests et la soutenance.

## 2. Nettoyage et normalisation

### Champs conservés

- `event_id`
- `title`
- `description`
- `conditions`
- `keywords`
- `city`
- `location_name`
- `address`
- `start_date`
- `end_date`
- `url`
- `image_url`
- `latitude`
- `longitude`
- `source`
- `attendance_mode`
- `status`

### Nettoyage appliqué

- suppression du HTML dans les champs texte ;
- normalisation des espaces ;
- normalisation des dates au format ISO ;
- extraction des coordonnées ;
- conservation d'un lien principal unique dans `url` ;
- suppression des doublons par `event_id` ;
- remplacement de certains champs manquants par des valeurs explicites.

### Champs volontairement exclus

- `age_min` et `age_max` : trop peu renseignés pour le POC ;
- `registration_url` et `external_url` : trop hétérogènes ;
- champs géographiques secondaires constants sur le périmètre choisi.

### Exemple court d'événement nettoyé

```json
{
  "event_id": "13708573",
  "title": "Initiation à l'astronomie à Lanton",
  "description": "Initiation à l'astronomie et observation aux instruments.. ✨ Découvrez le ciel nocturne comme vous ne l’avez jamais observé 🌙🔭 Lors de cette soirée d’observation du ciel , laissez-vous émerveiller par les beautés cachées de l’espace, visibles uniquement grâce aux télescopes et aux jumelles astronomiques . Planètes, galaxies, nébuleuses… vous découvrirez les plus beaux objets célestes du moment. Tout au long de la soirée, vous comprendrez : le parcours de vie d’une étoile , de sa naissance à sa fin les grands principes de la mécanique céleste et le déplacement apparent des astres Vous apprendrez également à : reconnaître plusieurs constellations , et peut-être celle de votre signe zodiacal si elle est visible découvrir les mythes et légendes associés aux constellations Des expériences simples et ludiques viendront accompagner les explications pour mieux comprendre le fonctionnement de notre ciel. Ce sera aussi l’occasion de : découvrir comment fonctionne un télescope poser toutes vos questions si vous envisagez d’en acquérir un, afin de faire un choix éclairé Dans une ambiance conviviale et un cadre naturel inhabituel, cette soirée est une invitation à lever les yeux vers les étoiles , à se détendre et à partager un moment hors du temps. Durée : 2h30 Activité organisée par Arnaud - Animateur scientifique Infos et réservation sur EcoNature : https://www.eco-nature.org/experience/initiation-a-l-astronomie-a-lanton-3725. Conditions : Tarifs : Enfant -18 ans 10€ ; Adulte 18 ans et + 20€",
  "conditions": "Tarifs : Enfant -18 ans 10€ ; Adulte 18 ans et + 20€",
  "city": "Lanton",
  "location_name": "Blagon, 33138 Lanton, France",
  "address": "Blagon, 33138 Lanton, France",
  "start_date": "2026-01-05T19:30:00+00:00",
  "end_date": "2027-01-03T21:30:00+00:00",
  "keywords": "Nature, Sortie nature, astronomie",
  "url": "https://openagenda.com/econature/events/initiation-a-lastronomie-a-lanton",
  "image_url": "https://cdn.openagenda.com/main/7d4acecf130c4c8cae6f20fcc5c64d56.base.image.jpg",
  "latitude": 44.783281,
  "longitude": -0.934394,
  "source": "EcoNature",
  "attendance_mode": "Sur place",
  "status": "Programmé"
}
```

## 3. Construction des documents textuels

### Objectif

Le fichier `events_clean.json` reste structuré. Le fichier `events_documents.jsonl` ajoute une représentation textuelle pensée pour le futur chunking et l'indexation FAISS.

### Différence entre données structurées, `document_text` et `metadata`

- les données structurées servent au nettoyage, aux contrôles qualité et aux transformations ultérieures ;
- `document_text` est une version lisible, compacte et utile pour la recherche sémantique ;
- `metadata` conserve les champs importants séparément pour le traçage, l'affichage et le filtrage.

### Champs utilisés dans `document_text`

Le texte peut inclure :

- titre ;
- description ;
- lieu ;
- ville ;
- adresse ;
- dates ;
- mots-clés ;
- conditions ;
- mode de participation ;
- statut ;
- source.

Les champs vides ne sont pas ajoutés.

L’URL n’est pas ajoutée au `document_text`, car elle n’apporte pas d’information sémantique utile pour la recherche. Elle reste conservée dans les `metadata` afin de pouvoir afficher la source ou rediriger l’utilisateur après récupération du document.

### Métadonnées conservées

- `event_id`
- `title`
- `city`
- `location_name`
- `start_date`
- `end_date`
- `url`
- `image_url`
- `latitude`
- `longitude`
- `source`

Les champs `attendance_mode` et `status` sont aussi conservés s'ils sont présents.

### Exemple court de `document_text` Markdown

```markdown
# Initiation à l'astronomie à Lanton

## Description
Initiation à l'astronomie et observation aux instruments.. ✨ Découvrez le ciel nocturne comme vous ne l’avez jamais observé 🌙🔭 Lors de cette soirée d’observation du ciel , laissez-vous émerveiller par les beautés cachées de l’espace, visibles uniquement grâce aux télescopes et aux jumelles astronomiques . Planètes, galaxies, nébuleuses… vous découvrirez les plus beaux objets célestes du moment. Tout au long de la soirée, vous comprendrez : le parcours de vie d’une étoile , de sa naissance à sa fin les grands principes de la mécanique céleste et le déplacement apparent des astres Vous apprendrez également à : reconnaître plusieurs constellations , et peut-être celle de votre signe zodiacal si elle est visible découvrir les mythes et légendes associés aux constellations Des expériences simples et ludiques viendront accompagner les explications pour mieux comprendre le fonctionnement de notre ciel. Ce sera aussi l’occasion de : découvrir comment fonctionne un télescope poser toutes vos questions si vous envisagez d’en acquérir un, afin de faire un choix éclairé Dans une ambiance conviviale et un cadre naturel inhabituel, cette soirée est une invitation à lever les yeux vers les étoiles , à se détendre et à partager un moment hors du temps. Durée : 2h30 Activité organisée par Arnaud - Animateur scientifique Infos et réservation sur EcoNature : https://www.eco-nature.org/experience/initiation-a-l-astronomie-a-lanton-3725. Conditions : Tarifs : Enfant -18 ans 10€ ; Adulte 18 ans et + 20€

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

Le fichier JSONL produit à cette étape est destiné au futur chunking puis à l'indexation FAISS.
