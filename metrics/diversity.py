import math
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist

def diversity_bkp(itens, pred_cat):
    entropy = []

    for item, cat in zip(itens, pred_cat):
        for alt, cat_alt in zip(itens, pred_cat):
            if item == alt: continue
            if cat == cat_alt:
                entropy.append(1)
            else:
                entropy.append(0)

    entropy_sum = sum(entropy)

    if entropy_sum == 0:
        return 1
    
    entropy = math.log2(sum(entropy))

    return 1 - (entropy / len(itens))

def diversity(top_recommends: dict, embeddings: np.ndarray, similarity_metric: str = 'cosine') -> float:

    diversity = 0
    for user, items in top_recommends.items():
        items_emb = embeddings[list(items.keys())]
        sim = pdist(items_emb, similarity_metric)
        user_diversity = 1 - np.mean(sim)
        diversity += user_diversity
    

    return diversity / len(top_recommends)
