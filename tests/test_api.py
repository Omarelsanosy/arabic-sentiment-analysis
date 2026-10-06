import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_sentiment_analysis.service import SentimentService


def test_service_predict_returns_a_list_of_results() -> None:
    service = SentimentService()

    results = service.predict(["الخدمة سيئة جدًا", "شحن سريع"])

    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert set(result.keys()) == {"sentiment", "confidence"}
        assert result["sentiment"] in {"Negative", "Neutral", "Positive"}
        assert 0.0 < result["confidence"] <= 1.0
