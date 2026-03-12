from __future__ import annotations

from src.utils.paths import PROCESSED_DATA_DIR, PREDICTION_DIR
from src.utils.io import read_csv, write_csv, write_json
from src.recommend.cbf import recommend_by_cbf
from src.recommend.ncf import run_neumf_random_search, fit_neumf, predict_ncf_scores


def run() -> None:
    res_matrix = read_csv(PROCESSED_DATA_DIR / "res_matrix.csv").set_index("p_id")
    user_matrix = read_csv(PROCESSED_DATA_DIR / "user_matrix.csv").set_index("u_id")
    score_board = read_csv(PROCESSED_DATA_DIR / "score_board.csv")

    best_params = run_neumf_random_search(score_board, top_k=10, seed=123)
    write_json(best_params, PREDICTION_DIR / "best_neumf_params.json")

    model, _ = fit_neumf(score_board, best_params, seed=123)
    ncf_scores = predict_ncf_scores(model, score_board)
    write_csv(ncf_scores, PREDICTION_DIR / "ncf_scores.csv")

    cbf_rows = []
    for user_id in user_matrix.index.tolist():
        rec_df = recommend_by_cbf(
            user_id=user_id,
            user_matrix=user_matrix,
            item_matrix=res_matrix,
            top_k=len(res_matrix),
        )
        rec_df["u_id"] = user_id
        cbf_rows.append(rec_df)

    cbf_scores = (
        read_csv(PREDICTION_DIR / "ncf_scores.csv")[["u_id", "p_id"]]
        .merge(
            __import__("pandas").concat(cbf_rows, ignore_index=True),
            on=["u_id", "p_id"],
            how="left",
        )
        .fillna(0.0)
    )

    write_csv(cbf_scores, PREDICTION_DIR / "cbf_scores.csv")


if __name__ == "__main__":
    run()