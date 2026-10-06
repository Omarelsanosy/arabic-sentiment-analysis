import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_sentiment_analysis.predict import predict_sentiment


def test_predict_sentiment_returns_schema_for_negative_review() -> None:
    result = predict_sentiment("الخدمة سيئة جدًا")

    assert set(result.keys()) == {"sentiment", "confidence"}
    assert result["sentiment"] == "Negative"
    assert 0.0 < result["confidence"] <= 1.0


def test_predict_sentiment_returns_positive_label_for_positive_review() -> None:
    result = predict_sentiment("شحن سريع")

    assert result["sentiment"] == "Positive"
    assert 0.0 < result["confidence"] <= 1.0
