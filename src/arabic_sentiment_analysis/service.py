from pathlib import Path

import bentoml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arabic_sentiment_analysis.preprocessing import clean_arabic_text
from arabic_sentiment_analysis.vocabulary_drift import (
    VocabularyPSI,
    load_reference_tokens,
)


@bentoml.service
class SentimentService:
    def __init__(self) -> None:
        model_path = (
            Path(__file__).resolve().parents[2] / "models" / "arabert_sentiment"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        from prometheus_client import Gauge

        self.vocabulary_psi_gauge = Gauge(
            "arabic_sentiment_vocabulary_psi",
            "Population Stability Index of inference token frequencies versus "
            "the packaged review-corpus reference.",
            multiprocess_mode="max",
        )
        self.vocabulary_psi_gauge.set(float("nan"))
        reference_tokens = load_reference_tokens(
            model_path / "vocabulary_reference.json"
        )
        ignored_token_ids = set(self.tokenizer.all_special_ids)
        if self.tokenizer.unk_token_id is not None:
            ignored_token_ids.discard(self.tokenizer.unk_token_id)
        self.vocabulary_psi = VocabularyPSI(
            reference_tokens,
            self.vocabulary_psi_gauge,
            ignored_token_ids,
        )

    @bentoml.api(batchable=True, max_batch_size=16, max_latency_ms=100)
    def predict(self, text: list[str]) -> list[dict[str, str | float]]:
        if not text:
            return []

        cleaned_text = [clean_arabic_text(item) for item in text]
        result: list[dict[str, str | float]] = [
            {"sentiment": "Neutral", "confidence": 0.0} for _ in cleaned_text
        ]

        valid_indices = [idx for idx, item in enumerate(cleaned_text) if item]
        if not valid_indices:
            return result

        inputs = self.tokenizer(
            [cleaned_text[idx] for idx in valid_indices],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence, predicted_id = probabilities.max(dim=1)
        self.vocabulary_psi.observe(
            inputs["input_ids"].tolist(),
            inputs["attention_mask"].tolist(),
        )

        for position, label_id, score in zip(
            valid_indices,
            predicted_id.tolist(),
            confidence.tolist(),
            strict=True,
        ):
            result[position] = {
                "sentiment": self.model.config.id2label[label_id],
                "confidence": score,
            }
        return result
