from __future__ import annotations

from pathlib import Path
import gzip
import shutil
import pandas as pd
from gensim.models import KeyedVectors


def ensure_unzipped_word2vec(gz_path: str | Path, bin_path: str | Path) -> Path:
    gz_path = Path(gz_path)
    bin_path = Path(bin_path)

    if not bin_path.exists():
        with gzip.open(gz_path, "rb") as f_in, open(bin_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    return bin_path


def load_word2vec_model(bin_path: str | Path) -> KeyedVectors:
    return KeyedVectors.load_word2vec_format(str(bin_path), binary=True)


def collect_unique_tags(tag_series: pd.Series) -> list[str]:
    all_tags: set[str] = set()
    for tags in tag_series.fillna(""):
        for tag in str(tags).split(","):
            tag = tag.strip()
            if tag:
                all_tags.add(tag)
    return sorted(all_tags)


def build_tag_embedding_dataframe(
    unique_tags: list[str],
    w2v_model: KeyedVectors,
) -> pd.DataFrame:
    vectors = {}
    for word in unique_tags:
        if word in w2v_model.key_to_index:
            vectors[word] = w2v_model.get_vector(word)

    if not vectors:
        raise ValueError("No tags matched the Word2Vec vocabulary.")

    df = pd.DataFrame(vectors)
    return df