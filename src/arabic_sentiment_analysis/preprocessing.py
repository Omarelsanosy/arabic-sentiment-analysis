import re

def clean_arabic_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text