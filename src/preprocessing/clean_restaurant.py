from __future__ import annotations

import pandas as pd


RESTAURANT_COLUMNS = [
    "name",
    "address",
    "category",
    "main_mn",
    "price",
    "opng_tm",
    "rating",
    "rvw_cnt",
    "tags",
]


def clean_restaurant_dataframe(restaurant_df: pd.DataFrame) -> pd.DataFrame:
    df = restaurant_df.copy()

    available_cols = [col for col in RESTAURANT_COLUMNS if col in df.columns]
    df = df[available_cols].copy()

    if "name" not in df.columns:
        raise ValueError("Restaurant dataframe must contain 'name' column.")

    df = df[df["name"].notna()].copy()
    df["name"] = df["name"].astype(str).str.strip()

    if "address" in df.columns:
        df["address"] = df["address"].fillna("").astype(str).str.strip()

    # 기존 노트북은 name 기준 중복 제거였지만,
    # Github용으로는 name + address 기준으로 먼저 제거하고,
    # address가 비어 있으면 name 기준으로 한번 더 정리
    if "address" in df.columns:
        df = df.drop_duplicates(subset=["name", "address"], keep="first")
    df = df.drop_duplicates(subset=["name"], keep="first")

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    if "rvw_cnt" in df.columns:
        df["rvw_cnt"] = pd.to_numeric(df["rvw_cnt"], errors="coerce")

    df.reset_index(drop=True, inplace=True)
    return df