import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist

def diversity(top_recommends: dict, embeddings: np.ndarray, similarity_metric: str = 'cosine') -> float:

    diversity = 0
    for user, items in top_recommends.items():
        items_emb = embeddings[[int(k) for k in items.keys()]]
        sim = pdist(items_emb, similarity_metric)
        user_diversity = 1 - np.mean(sim)
        diversity += user_diversity
    

    return diversity / len(top_recommends)
