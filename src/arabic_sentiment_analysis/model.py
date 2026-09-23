import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/arabert_sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model = model.to(DEVICE)
model.eval()