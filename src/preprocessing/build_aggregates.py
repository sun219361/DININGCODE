from __future__ import annotations

import pandas as pd


def _concat_reviews(series: pd.Series) -> str:
    values = [str(v).strip() for v in series if isinstance(v, str) and v.strip()]
    return ". ".join(values)


def build_restaurant_review_aggregate(
    restaurant_df: pd.DataFrame,
    review_df: pd.DataFrame,
    review_col: str = "review_eng",
) -> pd.DataFrame:
    info_df = restaurant_df.copy()
    rv_df = review_df.copy()

    agg = (
        rv_df.groupby("p_id", as_index=False)[review_col]
        .agg(_concat_reviews)
        .rename(columns={review_col: "reviews"})
    )

    info_df = info_df.merge(agg, on="p_id", how="left")
    info_df["reviews"] = info_df["reviews"].fillna("")
    return info_df


def build_user_review_aggregate(
    review_df: pd.DataFrame,
    review_col: str = "review_eng",
) -> pd.DataFrame:
    rv_df = review_df.copy()

    user_df = (
        rv_df.groupby("u_id", as_index=False)[review_col]
        .agg(_concat_reviews)
        .rename(columns={review_col: "reviews"})
    )
    user_df["reviews"] = user_df["reviews"].fillna("")
    return user_df