"""Readable prose from every input kind the tool accepts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdown", ".mkd"}
SUPPORTED_SUFFIXES = MARKDOWN_SUFFIXES


@dataclass(frozen=True)
class Source:
    """One file's readable prose."""

    name: str
    text: str


# Fenced code, front matter, and link targets are not prose. Scoring them would
# measure the writer's toolchain rather than their writing.
_FRONT_MATTER = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.S)
_FENCED_CODE = re.compile(r"^[ \t]*(`{3,}|~{3,}).*?^[ \t]*\1[ \t]*$", re.M | re.S)
_UNCLOSED_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,}).*\Z", re.M | re.S)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)|!\[[^\]]*\]\[[^\]]*\]")
_INLINE_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_REFERENCE_LINK = re.compile(r"\[([^\]]*)\]\[[^\]]*\]")
_LINK_DEFINITION = re.compile(r"^[ \t]*\[[^\]]+\]:.*$", re.M)
_AUTOLINK = re.compile(r"<(?:https?|mailto):[^>]*>")
_BARE_URL = re.compile(r"https?://\S+")
_ATX_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]*", re.M)
_SETEXT_RULE = re.compile(r"^[ \t]*(?:={3,}|-{3,}|\*{3,}|_{3,})[ \t]*$", re.M)
_BLOCKQUOTE = re.compile(r"^[ \t]*>+[ \t]?", re.M)
_LIST_MARKER = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+", re.M)
_TABLE_DIVIDER = re.compile(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$", re.M)
_EMPHASIS = re.compile(r"[*_~`]+")
_FOOTNOTE = re.compile(r"\[\^[^\]]*\]")
_BLANK_LINES = re.compile(r"\n{3,}")


def markdown_to_prose(markdown: str) -> str:
    """Reduce Markdown to the prose a reader would actually read.

    Code blocks, front matter, link targets, and table rules are removed
    because they say nothing about how the prose was written; heading and link
    text is kept because it is prose.
    """
    text = _FRONT_MATTER.sub("", markdown)
    text = _FENCED_CODE.sub("\n", text)
    text = _UNCLOSED_FENCE.sub("\n", text)
    text = _HTML_COMMENT.sub(" ", text)
    text = _LINK_DEFINITION.sub("", text)
    text = _IMAGE.sub(" ", text)
    text = _INLINE_LINK.sub(r"\1", text)
    text = _REFERENCE_LINK.sub(r"\1", text)
    text = _FOOTNOTE.sub("", text)
    text = _AUTOLINK.sub(" ", text)
    text = _BARE_URL.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _TABLE_DIVIDER.sub("", text)
    text = _SETEXT_RULE.sub("", text)
    text = _ATX_HEADING.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = text.replace("|", " ")

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def load_source(path: Path, display: str | None = None) -> Source:
    """Read one Markdown file into readable prose."""
    name = display or str(path)
    if path.suffix.lower() not in MARKDOWN_SUFFIXES:
        raise ValueError(
            f"Unsupported file type: {name} "
            f"(expected {', '.join(sorted(SUPPORTED_SUFFIXES))})"
        )
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise ValueError(f"Invalid markdown: {name}: {error}") from error
    return Source(name=name, text=markdown_to_prose(raw))


def load_sources(paths: list[Path]) -> list[Source]:
    """Read every input, failing on the first unreadable one."""
    if not paths:
        raise ValueError("No input files given")
    return [load_source(path) for path in paths]
