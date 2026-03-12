from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def load_stop_words(path: str | Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {word.strip().lower() for word in f.read().split("/") if word.strip()}


def normalize_tag(word: str) -> str:
    word = re.sub(r"[,\.\!\:\?\~\\\";]", "", word)
    return word.lower().strip()


def extract_adjective_tags(text: str, stop_words: set[str]) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    adjectives = []
    for token, pos in tagged:
        token_norm = normalize_tag(token)
        if pos == "JJ" and token_norm and token_norm not in stop_words:
            adjectives.append(token_norm)

    return ",".join(adjectives)


def add_tag_column(
    df: pd.DataFrame,
    text_col: str,
    stop_words: set[str],
    output_col: str = "tags",
) -> pd.DataFrame:
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

    out = df.copy()
    tags = []

    for text in tqdm(out[text_col], desc=f"Extracting tags from {text_col}"):
        tags.append(extract_adjective_tags(text, stop_words))

    out[output_col] = tags
    return out