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
poetry run python scripts/filter_openagenda_events.py
```
