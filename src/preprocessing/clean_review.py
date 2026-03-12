from __future__ import annotations

import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder


REVIEW_COLUMNS = ["res_name", "user_name", "rating", "review"]


def normalize_review_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace(".", ". ")
    text = text.replace("!", "! ")
    text = text.replace("(", "( ")
    text = text.replace(")", ") ")
    text = text.replace("^", "")
    text = text.replace("*", "")
    text = text.replace("-", " ")
    text = text.replace("\n", ". ")
    text = text.replace('"', ". ")

    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_review_dataframe(review_df: pd.DataFrame) -> pd.DataFrame:
    df = review_df.copy()

    missing_cols = set(REVIEW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required review columns: {missing_cols}")

    df = df[REVIEW_COLUMNS].copy()
    df = df[df["user_name"].notna()].copy()
    df = df[df["res_name"].notna()].copy()
    df["review"] = df["review"].fillna("").map(normalize_review_text)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df.reset_index(drop=True, inplace=True)
    return df


def encode_ids(
    restaurant_df: pd.DataFrame,
    review_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder, LabelEncoder]:
    info_df = restaurant_df.copy()
    rv_df = review_df.copy()

    place_encoder = LabelEncoder()
    place_encoder.fit(info_df["name"])

    unseen_restaurants = set(rv_df["res_name"]) - set(info_df["name"])
    if unseen_restaurants:
        rv_df = rv_df[rv_df["res_name"].isin(info_df["name"])].copy()

    info_df["p_id"] = place_encoder.transform(info_df["name"])
    rv_df["p_id"] = place_encoder.transform(rv_df["res_name"])

    user_encoder = LabelEncoder()
    user_encoder.fit(rv_df["user_name"])
    rv_df["u_id"] = user_encoder.transform(rv_df["user_name"])

    return info_df, rv_df, place_encoder, user_encoder