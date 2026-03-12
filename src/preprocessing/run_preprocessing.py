from __future__ import annotations

from pathlib import Path
import pickle

from src.utils.paths import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.io import read_csv, write_csv
from src.preprocessing.clean_restaurant import clean_restaurant_dataframe
from src.preprocessing.clean_review import clean_review_dataframe, encode_ids
from src.preprocessing.translate import ReviewTranslator
from src.preprocessing.build_aggregates import (
    build_restaurant_review_aggregate,
    build_user_review_aggregate,
)


def run() -> None:
    restaurant_raw = read_csv(RAW_DATA_DIR / "DiningCode_df.csv")
    review_raw = read_csv(RAW_DATA_DIR / "DiningCode_review_df.csv")

    restaurant_clean = clean_restaurant_dataframe(restaurant_raw)
    review_clean = clean_review_dataframe(review_raw)

    restaurant_encoded, review_encoded, place_encoder, user_encoder = encode_ids(
        restaurant_df=restaurant_clean,
        review_df=review_clean,
    )

    translator = ReviewTranslator(src="ko", dest="en", sleep_sec=0.0)
    review_translated = translator.translate_dataframe(
        review_encoded,
        review_col="review",
        output_col="review_eng",
        overwrite=False,
    )

    restaurant_final = build_restaurant_review_aggregate(
        restaurant_df=restaurant_encoded,
        review_df=review_translated,
        review_col="review_eng",
    )

    user_review_agg = build_user_review_aggregate(
        review_df=review_translated,
        review_col="review_eng",
    )

    write_csv(restaurant_final, INTERIM_DATA_DIR / "translated_eat_info.csv")
    write_csv(review_translated, INTERIM_DATA_DIR / "translated_eat_review.csv")
    write_csv(user_review_agg, INTERIM_DATA_DIR / "eat_review_by_user.csv")

    with open(PROCESSED_DATA_DIR / "place_encoder.pkl", "wb") as f:
        pickle.dump(place_encoder, f)

    with open(PROCESSED_DATA_DIR / "user_encoder.pkl", "wb") as f:
        pickle.dump(user_encoder, f)


if __name__ == "__main__":
    run()