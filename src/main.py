from __future__ import annotations

import argparse

from src.crawling.run_crawling import run as run_crawling
from src.preprocessing.run_preprocessing import run as run_preprocessing
from src.features.run_features import run as run_features
from src.recommend.run_recommend import run as run_recommend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["crawl", "preprocess", "features", "recommend", "all"],
    )
    args = parser.parse_args()

    if args.stage == "crawl":
        run_crawling()
    elif args.stage == "preprocess":
        run_preprocessing()
    elif args.stage == "features":
        run_features()
    elif args.stage == "recommend":
        run_recommend()
    elif args.stage == "all":
        run_crawling()
        run_preprocessing()
        run_features()
        run_recommend()


if __name__ == "__main__":
    main()