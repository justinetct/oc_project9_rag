# Diagramme UML de composants — Écho

Diagramme UML de composants simplifié du système RAG Écho (Puls-Events). Quatre zones :

- **Application Écho** : API FastAPI, RagService, Retriever LangChain.
- **Index vectoriel** : FAISS (index `index.faiss` + `index.pkl`).
- **Pipeline de données** : OpenAgenda → Préparation & indexation → Mistral Embeddings.
- **Service de génération** : Mistral LLM.

Le SVG inline du rapport HTML est dans [`assets/architecture_uml_components.svg`](assets/architecture_uml_components.svg). Le bloc Mermaid ci-dessous est la source de vérité.

```mermaid
flowchart TB
    User(("Utilisateur"))

    subgraph echo["Application Écho"]
        direction LR
        api["API FastAPI<br/><i>HTTP /ask</i>"]
        rag["RagService<br/><i>orchestration RAG</i>"]
        retr["Retriever LangChain<br/><i>recherche k=5</i>"]
        api --> rag
        rag --> retr
    end

    subgraph gen["Service de génération"]
        llm["Mistral LLM<br/><i>génération</i>"]
    end

    subgraph idx["Index vectoriel"]
        faiss["FAISS<br/><i>index vectoriel</i>"]
    end

    subgraph pipe["Pipeline de données"]
        direction LR
        oa["OpenAgenda<br/><i>événements</i>"]
        prep["Préparation &amp; indexation<br/><i>JSONL → chunks</i>"]
        emb["Mistral Embeddings<br/><i>vectorisation</i>"]
        oa --> prep
        prep --> emb
    end

    User -. "question" .-> api
    api -. "réponse" .-> User
    api -. "utilise" .-> rag
    rag -.-> retr
    rag -. "prompt + contexte" .-> llm
    retr -. "recherche" .-> faiss
    prep -. "indexation" .-> faiss

    classDef component fill:#FFFFFF,stroke:#1F3D5C,color:#0E2A47;
    classDef external  fill:#FAF1DC,stroke:#A07A30,color:#0E2A47;
    classDef actor     fill:#FFFFFF,stroke:#0E2A47,color:#0E2A47;

    class api,rag,retr,faiss,prep component;
    class llm,oa,emb external;
    class User actor;

    style echo fill:#F8FAFB,stroke:#1F3D5C;
    style idx  fill:#F8FAFB,stroke:#1F3D5C;
    style pipe fill:#F8FAFB,stroke:#1F3D5C;
    style gen  fill:#F8FAFB,stroke:#A07A30;
```

## Notes

- **Docker Compose** et **Swagger /docs** ne figurent pas dans le diagramme principal : ils relèvent du déploiement local et de la documentation, et sont décrits séparément dans la section "Docker" du rapport et du README.
- Les flèches pointillées suivent la notation UML standard de **dépendance** (`«use»`).
- Les composants externes (OpenAgenda, Mistral LLM, Mistral Embeddings) sont représentés en beige pour les distinguer des composants applicatifs internes Écho.

## Export SVG

Le SVG inline du rapport HTML ([`assets/architecture_uml_components.svg`](assets/architecture_uml_components.svg)) a été produit à la main pour rester indépendant de toute toolchain Node. Pour le régénérer depuis le source Mermaid ci-dessus :

```bash
# Installation locale (hors projet, ne pas ajouter à pyproject)
npx -p @mermaid-js/mermaid-cli mmdc \
  -i docs/architecture_diagram.md \
  -o docs/assets/architecture_uml_components.svg \
  -b transparent

# PNG haute résolution pour les slides
npx -p @mermaid-js/mermaid-cli mmdc \
  -i docs/architecture_diagram.md \
  -o docs/assets/architecture_uml_components.png \
  -w 1600 -H 1400 -b white
```
