import torch

from arabic_sentiment_analysis.model import DEVICE, model, tokenizer
from arabic_sentiment_analysis.preprocessing import clean_arabic_text

# =========================
# 5. Prediction Function
# =========================

def predict_sentiment(text: str) -> dict[str, str | float]:
    # Clean the input text
    text = clean_arabic_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Move tensors to the same device as the model
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # Disable gradient calculation
    with torch.no_grad():

        outputs = model(**inputs)

    # Get the most likely class and its probability
    probabilities = torch.softmax(outputs.logits, dim=1)
    confidence, predicted_id = probabilities.max(dim=1)
    confidence = confidence.item()
    predicted_id = predicted_id.item()

    # Get label from model config
    predicted_label = model.config.id2label[predicted_id]

    return {
        "sentiment": predicted_label,
        "confidence": confidence,
    }


# =========================
# 6. Test
# =========================

if __name__ == "__main__":

    text = "الخدمة سيئة جدًا"

    result = predict_sentiment(text)

    print("Text:", text)
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])