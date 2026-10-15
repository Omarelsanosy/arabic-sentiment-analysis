from collections.abc import Sequence

import pytest

from arabic_sentiment_analysis.vocabulary_drift import (
    VocabularyPSI,
    calculate_vocabulary_psi,
)


class GaugeStub:
    def __init__(self) -> None:
        self.values: list[float] = []

    def set(self, value: float) -> None:
        self.values.append(value)


def test_vocabulary_psi_is_low_for_matching_token_distributions() -> None:
    reference = [1] * 900 + [2] * 100
    current = [1] * 900 + [2] * 100

    assert calculate_vocabulary_psi(reference, current) == pytest.approx(0.0)


def test_vocabulary_psi_exceeds_alert_threshold_for_shifted_distribution() -> None:
    reference = [1] * 900 + [2] * 100
    current = [1] * 500 + [2] * 500

    assert calculate_vocabulary_psi(reference, current) > 0.25


def test_observe_updates_gauge_once_window_has_enough_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "arabic_sentiment_analysis.vocabulary_drift.MIN_CURRENT_TOKENS", 3
    )
    monkeypatch.setattr(
        "arabic_sentiment_analysis.vocabulary_drift.REPORT_INTERVAL_TOKENS", 3
    )
    gauge = GaugeStub()
    monitor = VocabularyPSI([1, 1, 1, 2], gauge, ignored_token_ids={0})
    rows: Sequence[Sequence[int]] = [[1, 1, 2, 0]]
    masks: Sequence[Sequence[int]] = [[1, 1, 1, 1]]

    monitor.observe(rows, masks)

    expected_psi = calculate_vocabulary_psi([1, 1, 1, 2], [1, 1, 2])
    assert gauge.values == pytest.approx([expected_psi])
