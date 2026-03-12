from __future__ import annotations

import time
import pandas as pd
from googletrans import Translator
from tqdm import tqdm


class ReviewTranslator:
    def __init__(self, src: str = "ko", dest: str = "en", sleep_sec: float = 0.0):
        self.src = src
        self.dest = dest
        self.sleep_sec = sleep_sec
        self.translator = Translator()

    def translate_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        try:
            translated = self.translator.translate(text, src=self.src, dest=self.dest)
            if self.sleep_sec > 0:
                time.sleep(self.sleep_sec)
            return translated.text
        except Exception:
            return ""

    def translate_dataframe(
        self,
        review_df: pd.DataFrame,
        review_col: str = "review",
        output_col: str = "review_eng",
        overwrite: bool = False,
    ) -> pd.DataFrame:
        df = review_df.copy()

        if output_col not in df.columns:
            df[output_col] = ""

        for idx in tqdm(df.index, desc="Translating reviews"):
            current_value = df.at[idx, output_col]
            if not overwrite and isinstance(current_value, str) and current_value.strip():
                continue

            review = df.at[idx, review_col]
            df.at[idx, output_col] = self.translate_text(review)

        return df