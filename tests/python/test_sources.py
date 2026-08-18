from pathlib import Path

import pytest

from slop_detector.models import chunk_sources
from slop_detector.sources import Source, load_source, load_sources, markdown_to_prose


FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_markdown_keeps_prose_and_drops_the_toolchain():
    prose = markdown_to_prose(
        "---\ntitle: Draft\nauthor: nobody\n---\n"
        "# A Heading\n\n"
        "Some real prose here.\n\n"
        "```python\ndef leaked(): return 'code'\n```\n\n"
        "More prose after the code.\n"
    )

    assert "A Heading" in prose
    assert "Some real prose here." in prose
    assert "More prose after the code." in prose
    assert "leaked" not in prose
    assert "title: Draft" not in prose


def test_markdown_keeps_link_text_but_not_link_targets():
    prose = markdown_to_prose(
        "See [the manual](https://example.com/manual) and [ref][1].\n\n"
        "![a diagram](diagram.png)\n\n"
        "[1]: https://example.com/other\n"
    )

    assert "See the manual and ref." in prose
    assert "example.com" not in prose
    assert "diagram" not in prose


def test_markdown_strips_structure_that_is_not_words():
    prose = markdown_to_prose(
        "> A quotation.\n\n"
        "- First item\n"
        "- Second item\n\n"
        "1. Numbered\n\n"
        "| Column | Other |\n|---|---|\n| a | b |\n\n"
        "***\n\n"
        "**bold** and _italic_ and `code`.\n"
    )

    assert "A quotation." in prose
    assert "First item" in prose
    assert "Numbered" in prose
    assert "bold and italic and code." in prose
    assert "|" not in prose
    assert "***" not in prose


def test_markdown_survives_an_unclosed_code_fence():
    prose = markdown_to_prose("Real prose.\n\n```\nnever closed\n")

    assert "Real prose." in prose
    assert "never closed" not in prose


def test_load_source_reads_markdown_and_reports_no_embedded_images(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Title\n\nA paragraph.\n")

    source = load_source(path)

    assert source.text == "Title\n\nA paragraph."
    assert source.images == []


def test_load_source_still_reads_webarchives():
    source = load_source(FIXTURES / "article-with-image.webarchive")

    assert source.text == "Headline\nFirst paragraph.\nSecond paragraph."
    assert len(source.images) == 1


def test_load_source_names_the_unsupported_type(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<html></html>")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_source(path)


def test_load_sources_rejects_an_empty_selection():
    with pytest.raises(ValueError, match="No input files"):
        load_sources([])


def test_chunks_never_span_two_files():
    sources = [
        Source("a.md", "Alpha one. Alpha two."),
        Source("b.md", "Beta one."),
    ]

    chunks = chunk_sources(sources)

    assert [chunk.source for chunk in chunks] == ["a.md", "b.md"]
    assert all(chunk.text for chunk in chunks)
    assert "Beta" not in chunks[0].text


def test_a_long_file_contributes_several_chunks_all_attributed_to_it():
    sources = [Source("long.md", "A sentence about things. " * 400)]

    chunks = chunk_sources(sources)

    assert len(chunks) > 1
    assert {chunk.source for chunk in chunks} == {"long.md"}
