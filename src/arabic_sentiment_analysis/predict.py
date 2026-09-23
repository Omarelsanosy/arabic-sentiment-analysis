import torch
from arabic_sentiment_analysis.model import tokenizer, model, DEVICE
from arabic_sentiment_analysis.preprocessing import clean_arabic_text

# =========================
# 5. Prediction Function
# =========================

def predict_sentiment(text: str) -> str:
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

    # Get class with highest probability
    predicted_id = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    # Get label from model config
    predicted_label = model.config.id2label[predicted_id]

    return predicted_label


# =========================
# 6. Test
# =========================

if __name__ == "__main__":

    text = "الخدمة سيئة جدًا"

    result = predict_sentiment(text)

    print("Text:", text)
    print("Sentiment:", result)