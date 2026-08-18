from pathlib import Path

import pytest

from slop_detector.webarchive import parse_webarchive


FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_parse_webarchive_returns_readable_body_text_and_embedded_raster_images():
    content = parse_webarchive(FIXTURES / "article-with-image.webarchive")

    assert content.text == "Headline\nFirst paragraph.\nSecond paragraph."
    assert [(image.index, image.mime_type) for image in content.images] == [
        (0, "image/png")
    ]


def test_parse_webarchive_rejects_non_plist_input():
    with pytest.raises(ValueError, match=r"^Invalid webarchive:"):
        parse_webarchive(FIXTURES / "malformed.webarchive")
