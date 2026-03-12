from __future__ import annotations

import numpy as np
import pandas as pd


def cosine_similarity_vector(user_vec: np.ndarray, item_matrix: np.ndarray) -> np.ndarray:
    user_norm = np.linalg.norm(user_vec)
    item_norm = np.linalg.norm(item_matrix, axis=1)

    if user_norm == 0:
        return np.zeros(item_matrix.shape[0])

    denom = user_norm * item_norm
    denom = np.where(denom == 0, 1e-12, denom)

    return (item_matrix @ user_vec) / denom


def recommend_by_cbf(
    user_id: int,
    user_matrix: pd.DataFrame,
    item_matrix: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    if user_id not in user_matrix.index:
        raise KeyError(f"user_id={user_id} not found in user_matrix")

    user_vec = user_matrix.loc[user_id].to_numpy(dtype=float)
    item_ids = item_matrix.index.to_list()
    item_values = item_matrix.to_numpy(dtype=float)

    scores = cosine_similarity_vector(user_vec, item_values)

    result = pd.DataFrame({
        "p_id": item_ids,
        "cbf_score": scores,
    }).sort_values("cbf_score", ascending=False)

    return result.head(top_k).reset_index(drop=True)