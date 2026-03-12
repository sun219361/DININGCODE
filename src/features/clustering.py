from __future__ import annotations

from collections import Counter
import pandas as pd
from sklearn.cluster import KMeans


def fit_kmeans_on_tag_vectors(
    tag_vector_df: pd.DataFrame,
    n_clusters: int = 120,
    random_state: int = 42,
) -> tuple[KMeans, pd.DataFrame]:
    x = tag_vector_df.T
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_ids = model.fit_predict(x)

    cluster_df = x.copy()
    cluster_df["kmeans_id"] = cluster_ids
    return model, cluster_df


def build_cluster_word_table(cluster_df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    out = pd.DataFrame()
    for cluster_id in range(n_clusters):
        words = list(cluster_df[cluster_df["kmeans_id"] == cluster_id].index)
        out[f"group_{cluster_id}"] = pd.Series(words)
    return out


def _build_group_lookup(cluster_word_df: pd.DataFrame) -> dict[str, str]:
    lookup = {}
    for col in cluster_word_df.columns:
        for word in cluster_word_df[col].dropna().astype(str):
            if word and word != "0":
                lookup[word] = col
    return lookup


def build_restaurant_matrix(
    restaurant_df: pd.DataFrame,
    cluster_word_df: pd.DataFrame,
    tag_col: str = "tags",
    id_col: str = "p_id",
) -> pd.DataFrame:
    group_lookup = _build_group_lookup(cluster_word_df)
    groups = list(cluster_word_df.columns)

    matrix = pd.DataFrame(0, index=restaurant_df[id_col], columns=groups, dtype=int)

    for _, row in restaurant_df.iterrows():
        item_id = row[id_col]
        tags = [tag.strip() for tag in str(row.get(tag_col, "")).split(",") if tag.strip()]
        tag_counts = Counter(tags)

        for tag, count in tag_counts.items():
            group = group_lookup.get(tag)
            if group is not None:
                matrix.at[item_id, group] += count

    matrix.index.name = id_col
    return matrix


def build_user_matrix(
    user_review_df: pd.DataFrame,
    cluster_word_df: pd.DataFrame,
    tag_col: str = "tags",
    id_col: str = "u_id",
) -> pd.DataFrame:
    group_lookup = _build_group_lookup(cluster_word_df)
    groups = list(cluster_word_df.columns)

    matrix = pd.DataFrame(0, index=user_review_df[id_col], columns=groups, dtype=int)

    for _, row in user_review_df.iterrows():
        user_id = row[id_col]
        tags = [tag.strip() for tag in str(row.get(tag_col, "")).split(",") if tag.strip()]
        tag_counts = Counter(tags)

        for tag, count in tag_counts.items():
            group = group_lookup.get(tag)
            if group is not None:
                matrix.at[user_id, group] += count

    matrix.index.name = id_col
    return matrix


def build_score_board(review_df: pd.DataFrame) -> pd.DataFrame:
    score_df = review_df[["u_id", "p_id", "rating"]].copy()
    score_df["rating"] = pd.to_numeric(score_df["rating"], errors="coerce")
    score_df = score_df.dropna(subset=["rating"])
    score_df = score_df.sort_values(by=["u_id", "p_id"]).reset_index(drop=True)
    return score_df