# Exploration OpenAgenda

## Objectif

Cette exploration vise à comprendre comment récupérer les événements depuis OpenAgenda avant de développer le client final.

Les objectifs sont :

- identifier l’endpoint à utiliser ;
- vérifier si une clé API est nécessaire ;
- tester les filtres de date et de localisation ;
- comprendre la pagination ;
- inspecter la structure JSON retournée ;
- identifier les champs utiles pour le futur système RAG ;
- sauvegarder un petit échantillon brut pour analyse.

L’exploration est réalisée dans le notebook :

```text
notebooks/01_openagenda_exploration.ipynb
```

## Endpoint testé

Base URL :

```text
https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records
```

Dataset :

```text
evenements-publics-openagenda
```

Aucune clé API n’est nécessaire pour les premiers appels publics testés.

## Paramètres utiles

- `limit` : nombre d’événements retournés par appel.
- `offset` : décalage utilisé pour paginer les résultats.
- `where` : filtre principal pour les dates, les villes et les conditions complexes.
- `order_by` : tri des résultats, par exemple `firstdate_begin asc`.
- `select` : sélection éventuelle des champs utiles.
- `refine` : filtre de facette, utile pour explorer rapidement dans l’interface, mais moins adapté aux règles métier complexes.

## Filtre temporel retenu

L’énoncé demande de cibler des événements récents, c’est-à-dire de moins d’un an.

Le filtrage ne doit pas être fait avec `refine`, qui sert surtout aux facettes. Le filtre retenu utilise donc `where` :

```text
firstdate_begin >= now(years=-1)
```

Ce filtre conserve les événements dont la première date est située dans les 12 derniers mois.

Pour un assistant de recommandation d’événements, ce choix reste pertinent car il permet de conserver des événements récents ou à venir, tout en évitant les événements trop anciens.

## Filtre géographique

Le périmètre retenu est volontairement local : Bassin d’Arcachon, Val de l’Eyre et communes proches.

Le filtre combine :

- le département : `Gironde` ;
- la région : `Nouvelle-Aquitaine` ;
- les codes pays : `FR` et `fr`, car les deux valeurs existent dans le dataset ;
- une liste de villes ciblées.

La différence observée entre le notebook et la console OpenDataSoft venait notamment du filtre pays : la console utilisait `FR` et `fr`, alors que la première version du code ne gardait que `FR`.

## Villes retenues

Liste actuelle :

- Arcachon
- La Teste-de-Buch
- Pyla-sur-Mer
- Gujan-Mestras
- Le Teich
- Biganos
- Audenge
- Lanton
- Andernos-les-Bains
- Arès
- Lège-Cap-Ferret
- Mios
- Marcheprime
- Salles
- Belin-Béliet
- Le Barp
- Lugos
- Saint-Magne

## Filtre complet

Le filtre complet est construit dans le code à partir de la configuration centralisée.

Exemple de filtre attendu :

```text
location_department = "Gironde"
AND location_region = "Nouvelle-Aquitaine"
AND location_countrycode IN ("FR", "fr")
AND location_city IN ("Arcachon", "La Teste-de-Buch", "Pyla-sur-Mer", "Gujan-Mestras", "Le Teich", "Biganos", "Audenge", "Lanton", "Andernos-les-Bains", "Arès", "Lège-Cap-Ferret", "Mios", "Marcheprime", "Salles", "Belin-Béliet", "Le Barp", "Lugos", "Saint-Magne")
AND firstdate_begin >= now(years=-1)
```

Avec ces filtres, le notebook retourne actuellement :

```text
Nombre total d'événements trouvés : 603
```

## Configuration centralisée

Une configuration centralisée a été ajoutée dans :

```text
src/config.py
```

Elle contient notamment :

- les chemins du projet ;
- l’endpoint OpenAgenda ;
- le dataset id ;
- les villes ciblées ;
- les codes pays à conserver ;
- le filtre temporel ;
- le tri ;
- la taille des pages ;
- la liste des champs utiles.

La logique spécifique à OpenAgenda pourra être déplacée dans :

```text
src/openagenda.py
```

Cela permet de garder `src/config.py` pour les constantes et les chemins, et de placer la logique de construction des requêtes dans un module dédié.

## Champs disponibles

Un exemple de réponse JSON a été inspecté dans le notebook.

Champs observés dans l’échantillon :

- `uid`
- `slug`
- `canonicalurl`
- `title_fr`
- `description_fr`
- `longdescription_fr`
- `conditions_fr`
- `keywords_fr`
- `image`
- `thumbnail`
- `updatedat`
- `daterange_fr`
- `firstdate_begin`
- `firstdate_end`
- `lastdate_begin`
- `lastdate_end`
- `timings`
- `accessibility_label_fr`
- `location_coordinates`
- `location_name`
- `location_address`
- `location_postalcode`
- `location_city`
- `location_department`
- `location_region`
- `location_countrycode`
- `onlineaccesslink`
- `age_min`
- `age_max`
- `originagenda_title`
- `registration`
- `links`

## Champs retenus pour le projet

Champs utiles pour identifier, recommander ou expliquer un événement :

- `uid`
- `slug`
- `canonicalurl`
- `title_fr`
- `description_fr`
- `longdescription_fr`
- `conditions_fr`
- `keywords_fr`
- `daterange_fr`
- `firstdate_begin`
- `firstdate_end`
- `lastdate_begin`
- `lastdate_end`
- `timings`
- `location_name`
- `location_address`
- `location_postalcode`
- `location_city`
- `location_coordinates`
- `accessibility_label_fr`
- `age_min`
- `age_max`
- `registration`
- `onlineaccesslink`
- `originagenda_title`

Les champs suivants sont utiles comme métadonnées de contrôle, mais ils ne seront probablement pas intégrés au futur texte RAG car ils sont constants dans le périmètre choisi :

- `location_department`
- `location_region`
- `location_countrycode`

## Problèmes observés

- Certains champs sont souvent manquants : conditions, mots-clés, accessibilité, âge, inscription, lien en ligne.
- Les descriptions longues peuvent contenir du HTML.
- Certains champs structurés sont renvoyés sous forme de chaînes JSON, par exemple `timings`, `registration`, `status` ou `attendancemode`.
- Le dataset contient deux valeurs pour le code pays : `FR` et `fr`.
- La pagination devra être gérée avec `limit` et `offset`.
- Le filtre `refine` est utile pour l’exploration visuelle, mais le code doit privilégier `where` pour les règles métier.

## Décisions pour la suite

- Utiliser l’endpoint `/records` du dataset `evenements-publics-openagenda`.
- Ne pas utiliser de clé API pour le POC.
- Utiliser `where` plutôt que `refine` pour les filtres métier.
- Filtrer les événements récents avec `firstdate_begin >= now(years=-1)`.
- Conserver une zone géographique limitée au Bassin d’Arcachon, au Val de l’Eyre et aux communes proches.
- Conserver les codes pays `FR` et `fr`.
- Trier les événements par date croissante avec `order_by=firstdate_begin asc`.
- Centraliser les constantes dans `src/config.py`.
- Déplacer la logique OpenAgenda dans `src/openagenda.py` si elle dépasse la simple exploration.
- Sauvegarder un échantillon brut dans `data/raw/sample_openagenda.json`.
- À l’étape suivante, nettoyer les champs texte, supprimer le HTML et construire un champ `document_text` exploitable pour le RAG.
- Séparer les champs utilisés dans le texte indexé des métadonnées techniques.
