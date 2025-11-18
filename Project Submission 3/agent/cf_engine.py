import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def strip_iri(iri: str) -> str:
    if iri.startswith("http://www.wikidata.org/entity/"):
        return iri.split("/")[-1]
    return iri


class CFEngine:

    def __init__(self, user_csv, item_csv):
        # Load datasets
        self.user_ratings = pd.read_csv(user_csv)
        self.item_ratings = pd.read_csv(item_csv)

        # Convert item IRIs → QIDs
        self.user_ratings["item_id"] = self.user_ratings["item_id"].apply(strip_iri)
        self.item_ratings["item_id"] = self.item_ratings["item_id"].apply(strip_iri)

        # Pivot to user-item matrix
        self.user_item_matrix = self.user_ratings.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            fill_value=0
        )

        # All item_ids in matrix
        self.items = np.array(self.user_item_matrix.columns)

        # Fit KNN model (item-based CF)
        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model.fit(self.user_item_matrix.T.values)

    # -------------------------
    # Simple helper: return QIDs
    # -------------------------
    def similar_items(self, qid: str, k=10):
        if qid not in self.items:
            return []

        idx = np.where(self.items == qid)[0][0]

        distances, indices = self.model.kneighbors(
            self.user_item_matrix.T.values[idx].reshape(1, -1),
            n_neighbors=k + 1
        )

        return [self.items[i] for i in indices.flatten()[1:]]

    # -------------------------
    # Main API used by your recommendation engine
    # -------------------------
    def recommend_for_item(self, item_ids, exclude_list, top_k=50):


        # Convert IRIs → QIDs
        qids = [strip_iri(i) for i in item_ids]
        exclude_qids = {strip_iri(i) for i in exclude_list}

        scores = {}

        for q in qids:
            if q not in self.items:
                continue

            neighbors = self.similar_items(q, k=top_k)

            for nb in neighbors:
                if nb in exclude_qids:
                    continue

                scores[nb] = scores.get(nb, 0) + 1  # simple voting

        # Sort by score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Convert QID → full IRI
        results = [(f"http://www.wikidata.org/entity/{qid}", score) for qid, score in sorted_items]

        return results[:top_k]