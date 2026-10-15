from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from arabic_sentiment_analysis.preprocessing import clean_arabic_text
from arabic_sentiment_analysis.vocabulary_drift import REFERENCE_SAMPLE_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "arabert_sentiment"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "CompanyReviews.csv"


def build_reference(data_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(data_path, usecols=["review_description"])
    texts = [
        clean_arabic_text(text)
        for text in frame["review_description"].dropna().astype(str)
    ]
    texts = [text for text in texts if text]
    if not texts:
        raise ValueError(f"No non-empty reviews found in {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    ignored_token_ids = set(tokenizer.all_special_ids)
    if tokenizer.unk_token_id is not None:
        ignored_token_ids.discard(tokenizer.unk_token_id)

    frequencies: Counter[int] = Counter()
    for start in range(0, len(texts), 128):
        tokens = tokenizer(
            texts[start : start + 128],
            truncation=True,
            max_length=128,
            padding=False,
        )
        frequencies.update(
            int(token_id)
            for row in tokens["input_ids"]
            for token_id in row
            if int(token_id) not in ignored_token_ids
        )
    if not frequencies:
        raise ValueError(f"No tokenizer vocabulary tokens found in {data_path}")

    token_ids = np.fromiter(frequencies.keys(), dtype=np.int64)
    probabilities = np.fromiter(frequencies.values(), dtype=np.float64)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(42)
    reference_sample = rng.choice(
        token_ids,
        size=REFERENCE_SAMPLE_SIZE,
        replace=True,
        p=probabilities,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "source": data_path.name,
                "sample_size": REFERENCE_SAMPLE_SIZE,
                "token_ids": reference_sample.tolist(),
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {REFERENCE_SAMPLE_SIZE:,} reference tokens to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a model-tokenizer vocabulary PSI reference sample."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_PATH / "vocabulary_reference.json",
    )
    args = parser.parse_args()
    build_reference(args.data, args.output)


if __name__ == "__main__":
    main()
