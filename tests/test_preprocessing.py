import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_sentiment_analysis.preprocessing import clean_arabic_text


def test_clean_arabic_text_removes_urls_mentions_and_extra_spaces() -> None:
    text = "  @user مرحبا بالعالم https://example.com  كيف الحال?  "

    cleaned = clean_arabic_text(text)

    assert cleaned == "مرحبا بالعالم كيف الحال?"


def test_clean_arabic_text_handles_plain_text_without_changes() -> None:
    text = "الخدمة ممتازة"

    cleaned = clean_arabic_text(text)

    assert cleaned == "الخدمة ممتازة"


def test_clean_arabic_text_handles_none_and_blank_input() -> None:
    assert clean_arabic_text(None) == ""
    assert clean_arabic_text("   ") == ""


def test_clean_arabic_text_removes_urls_mentions_and_hashtags() -> None:
    text = "#ممتاز @user هذا رابط https://example.com و #جيد"

    cleaned = clean_arabic_text(text)

    assert cleaned == "ممتاز هذا رابط و جيد"
