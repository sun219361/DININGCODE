from __future__ import annotations

from src.utils.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR
from src.utils.io import read_csv, write_csv
from src.features.tag_extraction import load_stop_words, add_tag_column
from src.features.vectorize import (
    ensure_unzipped_word2vec,
    load_word2vec_model,
    collect_unique_tags,
    build_tag_embedding_dataframe,
)
from src.features.clustering import (
    fit_kmeans_on_tag_vectors,
    build_cluster_word_table,
    build_restaurant_matrix,
    build_user_matrix,
    build_score_board,
)


def run() -> None:
    eat_info = read_csv(INTERIM_DATA_DIR / "translated_eat_info.csv")
    eat_review = read_csv(INTERIM_DATA_DIR / "translated_eat_review.csv")
    eat_review_by_user = read_csv(INTERIM_DATA_DIR / "eat_review_by_user.csv")

    stop_words = load_stop_words(PROCESSED_DATA_DIR / "stop_words.txt")

    eat_info = add_tag_column(eat_info, text_col="reviews", stop_words=stop_words, output_col="tags")
    eat_review_by_user = add_tag_column(
        eat_review_by_user, text_col="reviews", stop_words=stop_words, output_col="tags"
    )

    write_csv(eat_info, PROCESSED_DATA_DIR / "eat_info_with_tags.csv")
    write_csv(eat_review_by_user, PROCESSED_DATA_DIR / "eat_review_by_user_with_tags.csv")

    bin_path = ensure_unzipped_word2vec(
        gz_path=MODEL_DIR / "GoogleNews-vectors-negative300.bin.gz",
        bin_path=MODEL_DIR / "GoogleNews-vectors-negative300.bin",
    )
    w2v = load_word2vec_model(bin_path)

    unique_tags = collect_unique_tags(eat_info["tags"])
    tag_vector_df = build_tag_embedding_dataframe(unique_tags, w2v)

    n_clusters = 120
    _, cluster_df = fit_kmeans_on_tag_vectors(tag_vector_df, n_clusters=n_clusters)
    cluster_word_df = build_cluster_word_table(cluster_df, n_clusters=n_clusters)

    res_matrix = build_restaurant_matrix(eat_info, cluster_word_df, tag_col="tags", id_col="p_id")
    user_matrix = build_user_matrix(eat_review_by_user, cluster_word_df, tag_col="tags", id_col="u_id")
    score_board = build_score_board(eat_review)

    write_csv(cluster_word_df, PROCESSED_DATA_DIR / "eat_cluster.csv")
    write_csv(res_matrix.reset_index(), PROCESSED_DATA_DIR / "res_matrix.csv")
    write_csv(user_matrix.reset_index(), PROCESSED_DATA_DIR / "user_matrix.csv")
    write_csv(score_board, PROCESSED_DATA_DIR / "score_board.csv")


if __name__ == "__main__":
    run()