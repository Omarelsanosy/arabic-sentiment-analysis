import re


def clean_arabic_text(text: object) -> str:
    if text is None:
        return ""

    text = str(text)

    # Remove zero-width/invisible spaces that commonly appear in copied text.
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    text = text.replace("\u200e", " ").replace("\u200f", " ")

    # Remove URLs and email-like links.
    text = re.sub(
        r"https?://\S+|www\.\S+|\b\w+@\w+\.\w+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove mentions and hashtags, while preserving the word content of hashtags.
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)

    # Normalize spaces while leaving punctuation intact for downstream model input.
    text = re.sub(r"\s+", " ", text).strip()
    return text