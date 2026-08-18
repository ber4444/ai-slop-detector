from __future__ import annotations

import math
import re


MAX_CHUNK_CHARACTERS = 6000

# Sentence endings and blank-line/paragraph breaks are the natural boundaries a
# chunk may be cut at before falling back to word and character splits.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def select_device(torch_module: object) -> str:
    """Return the accelerator to run on, preferring Apple MPS over the CPU."""
    try:
        available = bool(torch_module.backends.mps.is_available())
    except AttributeError:
        available = False
    return "mps" if available else "cpu"


def chunk_text(text: str, max_characters: int = MAX_CHUNK_CHARACTERS) -> list[str]:
    """Split text into model-safe chunks, preferring natural text boundaries."""
    if max_characters < 1:
        raise ValueError("max_characters must be at least 1")

    chunks: list[str] = []
    current = ""
    for segment in _BOUNDARY.split(text):
        for piece in _split_overlong(segment.strip(), max_characters):
            candidate = f"{current} {piece}" if current else piece
            if len(candidate) <= max_characters:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _split_overlong(segment: str, max_characters: int) -> list[str]:
    """Break a single segment that cannot fit, on words then on characters."""
    if not segment:
        return []
    if len(segment) <= max_characters:
        return [segment]

    pieces: list[str] = []
    current = ""
    for word in segment.split():
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_characters:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        while len(word) > max_characters:
            pieces.append(word[:max_characters])
            word = word[max_characters:]
        current = word
    if current:
        pieces.append(current)
    return pieces


def softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax over a plain list of logits."""
    if not logits:
        raise ValueError("softmax requires at least one logit")
    largest = max(logits)
    exponentials = [math.exp(value - largest) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def editlens_score(logits: list[float]) -> float:
    """Normalize EditLens bucket logits to an expected AI-edit extent in [0, 1]."""
    if len(logits) < 2:
        raise ValueError("EditLens requires at least two edit-extent buckets")
    probabilities = softmax(logits)
    expected = sum(index * value for index, value in enumerate(probabilities))
    return expected / (len(probabilities) - 1)


def mean_score(scores: list[float]) -> float:
    """Average chunk-level scores; no chunks means there is nothing to report."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
