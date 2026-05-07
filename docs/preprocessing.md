# Pré-processing temporel des événements OpenAgenda

## Objectif

Le pré-processing temporel sert à conserver, pour le corpus principal du RAG, les événements encore actifs ou à venir à partir de la date de référence du POC.

## Fichiers

- fichier d’entrée : `data/raw/openagenda_events_raw.json`
- fichier de sortie : `data/processed/events_filtered.json`

## Date de référence

La date de référence retenue pour le POC est :

```text
2026-05-01
```

## Règle de filtrage

La règle principale est :

```text
lastdate_end >= 2026-05-01
```

On conserve donc les événements dont la dernière date de fin est supérieure ou égale au 1er mai 2026.

## Fallback si `lastdate_end` est absent

Si `lastdate_end` est absent, le filtrage essaie dans cet ordre :

1. `firstdate_end`
2. `firstdate_begin`

Si aucune date exploitable n’est disponible, l’événement est exclu.

## Raisons du choix

### Raison métier

Ce filtrage évite d’indexer des événements déjà terminés et permet de garder un corpus plus utile pour un chatbot centré sur les événements culturels encore pertinents.

### Raison technique

La date figée rend le POC reproductible pour les tests, la soutenance et les comparaisons de résultats.

En production, cette date pourrait être remplacée par une date dynamique.

## Commande

```bash
poetry run python scripts/02_filter_openagenda_events.py
```

## Nettoyage des événements

### Fichiers

- fichier d’entrée : `data/processed/events_filtered.json`
- fichier de sortie : `data/processed/events_clean.json`

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

### Nettoyage effectué

- suppression du HTML dans les champs texte ;
- normalisation des espaces ;
- construction d’une description propre à partir de `description_fr` et `longdescription_fr` ;
- ajout des conditions dans la description sous la forme `Conditions : ...` quand elles existent ;
- conservation des conditions dans un champ séparé `conditions` ;
- normalisation des mots-clés dans `keywords` ;
- suppression des mots-clés techniques de type `challenge-id=...` ;
- normalisation des dates au format ISO ;
- conservation du lien principal dans `url` à partir de `canonicalurl`, puis `onlineaccesslink` ;
- sélection d'une image depuis `image`, `thumbnail` ou `originalimage` ;
- extraction des coordonnées depuis `location_coordinates` ;
- extraction du mode de participation dans `attendance_mode` ;
- extraction du statut dans `status` ;
- remplacement des champs textuels manquants par des valeurs explicites ;
- conservation des valeurs structurées manquantes avec `null` ;
- suppression des doublons par `event_id`.

### Champs volontairement exclus

Certains champs restent disponibles dans les données brutes, mais ne sont pas conservés dans le dataset nettoyé :

- `age_min` et `age_max` : trop rarement renseignés dans le snapshot du POC ;
- `registration_url` et `external_url` : valeurs trop hétérogènes pour le dataset nettoyé ;
- `location_department`, `location_region` et `country_fr` : constantes dans le périmètre géographique choisi.

Le dataset nettoyé reste structuré, mais le futur texte indexable pour le RAG sera encore plus sélectif : les champs vides ou peu informatifs ne seront pas ajoutés au `document_text`.
  
### Commande

```bash
poetry run python scripts/03_clean_openagenda_events.py
```
