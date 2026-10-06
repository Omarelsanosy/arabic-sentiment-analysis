from pathlib import Path

import bentoml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arabic_sentiment_analysis.preprocessing import clean_arabic_text


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
    
        
    @bentoml.api(batchable=True, max_batch_size=16, max_latency_ms=100)
    def predict(self, text: list[str]) -> list[dict[str, str | float]]:
        cleaned_text = [clean_arabic_text(item) for item in text]
        inputs = self.tokenizer(
            cleaned_text,
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
        return [
            {
                "sentiment": self.model.config.id2label[label_id],
                "confidence": score,
            }
            for label_id, score in zip(predicted_id.tolist(), confidence.tolist())
        ]
