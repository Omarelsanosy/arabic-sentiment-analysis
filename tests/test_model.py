import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_sentiment_analysis.model import DEVICE, model, tokenizer


def test_model_and_tokenizer_are_loaded() -> None:
    assert tokenizer is not None
    assert model is not None
    assert model.training is False
    assert model.num_labels == 3
    assert set(model.config.id2label.values()) == {"Negative", "Neutral", "Positive"}


def test_model_outputs_logits_with_three_classes() -> None:
    texts = ["الخدمة سيئة جدًا", "شحن سريع"]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs.logits.shape == (2, 3)
