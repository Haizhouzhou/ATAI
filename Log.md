# Final Project Execution Log: Data Architecture & Strategy

This document details the data pipeline, file schemas, and strategic decisions implemented for the ATAI Final Evaluation. It serves as a technical reference for reproducing the chatbot's environment.

---

## 1. Core Data Resources (Read-Only)

These files are provided by the course environment at `/space_mounts/atai-hs25/dataset`. We rely on them as the source of truth but do not modify them.

### A. `graph.nt` (Knowledge Graph)
* **Path:** `/space_mounts/atai-hs25/dataset/graph.nt`
* **Format:** N-Triples (RDF)
* **Schema Description:**
    Each line represents a triple `<Subject> <Predicate> <Object> .`.
    ```text
    [http://www.wikidata.org/entity/Q123](http://www.wikidata.org/entity/Q123) [http://www.wikidata.org/prop/direct/P57](http://www.wikidata.org/prop/direct/P57) [http://www.wikidata.org/entity/Q456](http://www.wikidata.org/entity/Q456) .
    [http://www.wikidata.org/entity/Q123](http://www.wikidata.org/entity/Q123) [http://www.w3.org/2000/01/rdf-schema#label](http://www.w3.org/2000/01/rdf-schema#label) "Movie Title"@en .
    ```
* **Usage:**
    * **SPARQL Endpoint:** Loaded into `rdflib.Graph` for factual QA (e.g., "Who directed Fargo?").
    * **Training Data:** Parsed to generate positive triples for TransE embedding training.
* **Known Issues:** Missing `P57` (Director) and `P495` (Country) for several entities mentioned in the evaluation (e.g., *The Longest Day*). Handled via Embedding Inference fallback.

### B. `entity_ids.del` & `relation_ids.del`
* **Path:** `/space_mounts/atai-hs25/dataset/embeddings/*.del`
* **Format:** Tab-Separated Values (TSV), no header.
* **Schema:** `Index \t URI`
    ```text
    0   [http://www.wikidata.org/entity/Q123](http://www.wikidata.org/entity/Q123)
    1   [http://www.wikidata.org/entity/Q456](http://www.wikidata.org/entity/Q456)
    ```
* **Constraint:** Our custom embeddings **must** align strictly with these indices. Row `N` in our `.npy` matrix corresponds to the URI at Index `N` in this file.

### C. `images.json` (Multimedia Index)
* **Path:** `/space_mounts/atai-hs25/dataset/additional/images.json`
* **Format:** JSON List of Objects.
* **Schema:**
    ```json
    [
      {
        "img": "0077/4bZr...jpg",   // Relative path to image
        "movie": ["tt0109830"],     // List of associated IMDb IDs (Movies)
        "cast": ["nm0000158"],      // List of associated IMDb IDs (Actors)
        "id": "tt0109830"           // Primary ID
      }
    ]
    ```
* **Integration Challenge:** This file uses **IMDb IDs** (`tt...`, `nm...`), while the Knowledge Graph uses **Wikidata URIs** (`Q...`).
* **Solution:** We generated `imdb_map.json` (see below) to bridge this gap using the `wdt:P345` property found in the graph.

---

## 2. Custom Generated Resources (Submission Artifacts)

These files were generated to solve specific data quality issues (degenerate teacher embeddings, missing labels, dirty ID formats). They are located in the project submission folder.

### A. `embeddings/RFC_entity_embeds.npy`
* **Description:** Custom trained TransE entity embeddings (Dimension: 256).
* **Why Retrain?**: The provided teacher embeddings had a cosine similarity of `1.0` for unrelated entities, rendering them useless for recommendation.
* **Generation Logic (Reproducible):**
    1.  Load `entity_ids.del` to establish the fixed index order.
    2.  Load triples from `graph.nt`.
    3.  Train TransE model (Margin Ranking Loss, L2 Norm).
    4.  Save weights as `.npy` array.

### B. `embeddings/RFC_relation_embeds.npy`
* **Description:** Custom trained TransE relation embeddings.
* **Usage:** Required for **Embedding Inference QA** ("I think...").
* **Logic:** `Prediction = Vector(Head Entity) + Vector(Relation)`. We search for the nearest Entity vector to this result.

### C. `metadata/entity_labels.json`
* **Description:** A pre-computed dictionary mapping URIs to human-readable labels.
* **Schema:** `{"http://www.wikidata.org/entity/Q123": "The Matrix"}`
* **Reason:** SPARQL queries for labels often failed due to missing language tags or graph sparsity. This lookup table ensures we never return "Unknown Entity" to the user.

### D. `metadata/imdb_map.json`
* **Description:** A cleaned mapping from Wikidata URI to IMDb ID.
* **Schema:** `{"http://www.wikidata.org/entity/Q123": "tt0133093"}`
* **Data Cleaning Performed:**
    * **Raw Data:** `"tt0133093"^^<http://www.w3.org/2001/XMLSchema#string>` (Unusable key).
    * **Cleaned Data:** `"tt0133093"` (Matches `images.json`).
* **Usage:** Used by `MultimediaIndex` to find images for entities identified by the chatbot.

---

## 3. Strategic Approaches to Evaluation Tasks

### A. One-Hop QA (Factual & Inference)
**Requirement:** Answer factoids (e.g., "Who directed Fargo?").
* **Approach:**
    1.  **Entity Linking:** Identify the subject (e.g., *Fargo* `Q222720`) using our `EntityLinker`.
    2.  **SPARQL Lookup:** Attempt to find the answer tuple in `graph.nt`.
        * *Result:* Returns accurate facts (e.g., "Ethan Coen").
    3.  **Embedding Fallback:** If SPARQL returns nothing (due to missing edges in the dataset), use `RFC_relation_embeds.npy` to predict the most likely answer.
        * *Result:* Returns "I think [Predicted Name]" (e.g., for *The Longest Day* where director data is missing).

### B. Recommendation System
**Requirement:** Recommend similar movies.
* **Approach:**
    1.  **Priority Entity Linking:** The linker prioritizes entities that have a `ddis:rating` property. This solves name collisions (e.g., choosing *Back to the Future* the **Movie** over the **Video Game**).
    2.  **Hybrid Filtering:**
        * **Graph:** Prefer movies sharing Director/Genre.
        * **Embeddings:** Use Cosine Similarity on `RFC_entity_embeds.npy`.
    3.  **Strict Type Check:** We implemented an `is_movie` filter to prevent the system from recommending Actors (e.g., Jennifer Connelly) when asked about movies.
    4.  **Fail-Open Safety:** If filters remove all candidates (e.g., input was an Actor with no movie neighbors), the system falls back to a curated list of **Top Rated Movies** to ensure a helpful response.

### C. Multimedia
**Requirement:** Show images/posters.
* **Approach:**
    1.  User asks "Show me Halle Berry".
    2.  Linker finds URI `Q1033016`.
    3.  `MultimediaIndex` looks up `Q1033016` in `imdb_map.json` -> gets `nm0000932`.
    4.  Looks up `nm0000932` in `images.json` -> gets `0344/knLB...jpg`.
    5.  Bot formats response: `image:0344/knLB...jpg`.
* **Missing Data Note:** Some entities (e.g., *Forrest Gump*) are correctly identified and linked to IMDb IDs, but their IDs are physically missing from the provided `images.json`. In these cases, the bot correctly reports "Image not found".