from __future__ import annotations

import pandas as pd


def min_max_scale(series: pd.Series) -> pd.Series:
    min_v = series.min()
    max_v = series.max()
    if max_v == min_v:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_v) / (max_v - min_v)


def build_hybrid_scores(
    cbf_df: pd.DataFrame,
    ncf_df: pd.DataFrame,
    cbf_weight: float = 0.4,
    ncf_weight: float = 0.6,
) -> pd.DataFrame:
    merged = cbf_df.merge(ncf_df, on="p_id", how="outer").fillna(0.0)

    merged["cbf_score_scaled"] = min_max_scale(merged["cbf_score"])
    merged["ncf_score_scaled"] = min_max_scale(merged["ncf_score"])

    merged["hybrid_score"] = (
        cbf_weight * merged["cbf_score_scaled"] +
        ncf_weight * merged["ncf_score_scaled"]
    )

    return merged.sort_values("hybrid_score", ascending=False).reset_index(drop=True)


def recommend_hybrid_for_user(
    user_id: int,
    cbf_scores_all: pd.DataFrame,
    ncf_scores_all: pd.DataFrame,
    top_k: int = 10,
    cbf_weight: float = 0.4,
    ncf_weight: float = 0.6,
) -> pd.DataFrame:
    cbf_user = cbf_scores_all.copy()
    ncf_user = ncf_scores_all[ncf_scores_all["u_id"] == user_id][["p_id", "ncf_score"]].copy()

    hybrid = build_hybrid_scores(
        cbf_df=cbf_user[["p_id", "cbf_score"]],
        ncf_df=ncf_user,
        cbf_weight=cbf_weight,
        ncf_weight=ncf_weight,
    )
    return hybrid.head(top_k)