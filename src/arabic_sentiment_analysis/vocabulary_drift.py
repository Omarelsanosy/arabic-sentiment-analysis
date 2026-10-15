from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import pandas as pd
from evidently import Report
from evidently.metrics import ValueDrift

REFERENCE_SAMPLE_SIZE = 50_000
CURRENT_WINDOW_SIZE = 50_000
MIN_CURRENT_TOKENS = 1_000
REPORT_INTERVAL_TOKENS = 1_000


class Gauge(Protocol):
    def set(self, value: float) -> None: ...


def load_reference_tokens(path: Path) -> list[int]:
    reference = json.loads(path.read_text(encoding="utf-8"))
    token_ids = reference.get("token_ids") if isinstance(reference, dict) else None
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError(f"Vocabulary PSI reference has no token_ids: {path}")
    if any(not isinstance(token_id, int) for token_id in token_ids):
        raise ValueError(f"Vocabulary PSI reference contains invalid token IDs: {path}")
    return token_ids


def calculate_vocabulary_psi(
    reference_token_ids: Sequence[int],
    current_token_ids: Sequence[int],
) -> float:
    if not reference_token_ids or not current_token_ids:
        raise ValueError("PSI requires non-empty reference and current token samples")

    report = Report(
        metrics=[ValueDrift(column="token_id", method="psi", threshold=0.25)]
    )
    snapshot = report.run(
        current_data=pd.DataFrame(
            {"token_id": pd.Series(current_token_ids, dtype="category")}
        ),
        reference_data=pd.DataFrame(
            {"token_id": pd.Series(reference_token_ids, dtype="category")}
        ),
    )
    metric_value = snapshot.dict()["metrics"][0]["value"]
    if metric_value is None:
        raise RuntimeError("Evidently did not return a vocabulary PSI value")
    return float(metric_value)


class VocabularyPSI:
    def __init__(
        self,
        reference_token_ids: Sequence[int],
        gauge: Gauge,
        ignored_token_ids: set[int],
    ) -> None:
        if not reference_token_ids:
            raise ValueError("Vocabulary PSI requires a non-empty reference sample")
        self._reference = list(reference_token_ids)
        self._gauge = gauge
        self._ignored_token_ids = ignored_token_ids
        self._current: deque[int] = deque(maxlen=CURRENT_WINDOW_SIZE)
        self._tokens_since_report = 0
        self._lock = threading.Lock()
        self._report_lock = threading.Lock()

    def observe(
        self,
        input_ids: Sequence[Sequence[int]],
        attention_mask: Sequence[Sequence[int]],
    ) -> None:
        if len(input_ids) != len(attention_mask):
            raise ValueError("input_ids and attention_mask must have matching batches")

        new_tokens = [
            int(token_id)
            for token_row, mask_row in zip(input_ids, attention_mask, strict=True)
            for token_id, is_active in zip(token_row, mask_row, strict=True)
            if is_active and int(token_id) not in self._ignored_token_ids
        ]
        if not new_tokens:
            return

        with self._lock:
            self._current.extend(new_tokens)
            self._tokens_since_report += len(new_tokens)
            enough_tokens = len(self._current) >= MIN_CURRENT_TOKENS
            should_report = enough_tokens and (
                self._tokens_since_report >= REPORT_INTERVAL_TOKENS
            )
            if should_report:
                current_snapshot = list(self._current)
                self._tokens_since_report = 0
            else:
                current_snapshot = []

        if current_snapshot:
            with self._report_lock:
                psi = calculate_vocabulary_psi(self._reference, current_snapshot)
                self._gauge.set(psi)
