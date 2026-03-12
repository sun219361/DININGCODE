from __future__ import annotations

import warnings
import pandas as pd
import cornac
from cornac.eval_methods import RatioSplit
from cornac.hyperopt import Continuous, Discrete, RandomSearch

warnings.filterwarnings("ignore")


def prepare_cornac_data(score_df: pd.DataFrame) -> list[tuple]:
    df = score_df.copy()
    df["rating"] = df["rating"].astype(float)
    df = df.rename(columns={"u_id": "userID", "p_id": "itemID"})
    return [tuple(row) for row in df[["userID", "itemID", "rating"]].values]


def run_neumf_random_search(
    score_df: pd.DataFrame,
    top_k: int = 10,
    seed: int = 123,
):
    data = prepare_cornac_data(score_df)

    ratio_split = RatioSplit(
        data=data,
        test_size=0.1,
        val_size=0.1,
        rating_threshold=1.0,
        seed=seed,
        verbose=True,
    )

    neumf = cornac.models.NeuMF(
        layers=[64, 32, 16, 8],
        act_fn="tanh",
        learner="adam",
        num_neg=50,
        seed=seed,
        early_stopping={"min_delta": 0.0, "patience": 5},
    )

    ndcg = cornac.metrics.NDCG(k=top_k)
    precision = cornac.metrics.Precision(k=top_k)
    recall = cornac.metrics.Recall(k=top_k)
    fmeasure = cornac.metrics.FMeasure(k=top_k)

    search = RandomSearch(
        model=neumf,
        space=[
            Discrete("num_epochs", [50, 100, 150, 200]),
            Discrete("num_factors", [4, 8]),
            Discrete("batch_size", [128, 256, 512]),
            Continuous("lr", 0.001, 0.01),
        ],
        metric=fmeasure,
        eval_method=ratio_split,
    )

    cornac.Experiment(
        eval_method=ratio_split,
        models=[search],
        metrics=[ndcg, precision, recall, fmeasure],
    ).run()

    return search.best_params


def fit_neumf(score_df: pd.DataFrame, best_params: dict, seed: int = 123):
    data = prepare_cornac_data(score_df)

    ratio_split = RatioSplit(
        data=data,
        test_size=0.0,
        rating_threshold=1.0,
        verbose=True,
    )

    model = cornac.models.NeuMF(
        num_factors=best_params["num_factors"],
        layers=[64, 32, 16, 8],
        act_fn="tanh",
        learner="adam",
        num_epochs=best_params["num_epochs"],
        lr=best_params["lr"],
        num_neg=50,
        batch_size=best_params["batch_size"],
        seed=seed,
        early_stopping={"min_delta": 0.0, "patience": 5},
    )

    cornac.Experiment(
        eval_method=ratio_split,
        models=[model],
        metrics=[],
    ).run()

    return model, ratio_split


def predict_ncf_scores(
    model,
    score_df: pd.DataFrame,
    target_user_ids: list[int] | None = None,
) -> pd.DataFrame:
    user_ids = sorted(score_df["u_id"].unique().tolist())
    item_ids = sorted(score_df["p_id"].unique().tolist())

    results = []

    for internal_uidx, user_id in enumerate(user_ids):
        if target_user_ids is not None and user_id not in target_user_ids:
            continue

        ranked = model.rank(internal_uidx)
        ranked_item_indices = ranked[0]
        ranked_scores = ranked[1]

        for item_idx, score in zip(ranked_item_indices, ranked_scores):
            results.append({
                "u_id": user_id,
                "p_id": item_ids[item_idx],
                "ncf_score": float(score),
            })

    return pd.DataFrame(results)