"""Worker entry point: one JSON report on stdout, everything else on stderr."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .contracts import RunReport, RunRequest
from .models import (
    EDITLENS_NAME,
    TransformersLoader,
    chunk_sources,
    compare_detectors,
    mean_score,
    run_image_detector,
    run_text_detectors,
    select_runtime_device,
)
from .sources import load_sources
from .webarchive import EmbeddedImage


TOKEN_RELATIVE_PATH = Path(".secrets/huggingface.token")

# Kept short and jargon-free: this is what someone reads on every run.
ADVISORY_NOTES = (
    "These are guesses, not proof. Detectors are wrong often enough that you "
    "should not rely on them for anything that matters.",
    "Each detector is shown separately on purpose. They are never combined "
    "into one answer, because they measure different things and often "
    "disagree.",
    "Human writing can read as AI-generated, especially formal, technical, or "
    "heavily edited text, and text by people writing in a second language.",
)

# The mechanics, for a reader who asked for them.
VERBOSE_NOTES = (
    "Percentages are each model's own confidence. Under 10% reads as almost "
    "certainly human, under 40% probably human, 60% or more probably "
    "AI-generated, 90% or more almost certainly so.",
    "EditLens estimates how much of the text was AI-edited. That is a "
    "different quantity from the others' probability that it was AI-written.",
    "Every detector scores the same chunks; each percentage is the mean over "
    "them, and the per-chunk spread is shown beside it.",
)

IMAGE_NOTES = (
    "The image detector was trained to spot one family of AI image generators. "
    "A low score does not mean an image is real.",
    "Metadata findings come from the file's own labelling, not from looking at "
    "the picture. Most images carry none, and that means nothing either way.",
)


def read_hf_token() -> str:
    """Return the Hugging Face token, read only from the gitignored secret file.

    The token value is never included in an error message.
    """
    for directory in [Path.cwd(), *Path.cwd().parents]:
        candidate = directory / TOKEN_RELATIVE_PATH
        if not candidate.is_file():
            continue
        try:
            token = candidate.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise RuntimeError(
                f"Hugging Face token unreadable at {TOKEN_RELATIVE_PATH}: "
                f"{type(error).__name__}"
            ) from None
        if token:
            return token
        break
    raise RuntimeError(
        f"Hugging Face token missing: place it in {TOKEN_RELATIVE_PATH}"
    )


def build_report(request: RunRequest) -> RunReport:
    """Read every input and run the requested detectors sequentially."""
    sources = load_sources([Path(path) for path in request.paths])
    chunks = chunk_sources(sources)
    images = _collect_images(sources)
    device, warnings = select_runtime_device()
    token = read_hf_token()

    text = run_text_detectors(
        chunks, TransformersLoader(token), device, request.verbose
    )
    if text and all(result.score is None for result in text):
        raise RuntimeError(
            "No text detector could run:\n"
            + "\n".join(f"{result.name}: {result.detail}" for result in text)
        )
    image_results = (
        run_image_detector(images, TransformersLoader(token), device)
        if request.include_images
        else []
    )

    notes = list(warnings)
    notes.extend(_unavailable_notes(text, image_results))
    notes.extend(_scope_notes(sources, chunks))
    notes.extend(_agreement_notes(text, request.verbose))
    notes.extend(ADVISORY_NOTES)
    if request.include_images:
        notes.extend(IMAGE_NOTES)
    if request.verbose:
        notes.extend(VERBOSE_NOTES)
        notes.extend(_per_file_notes(chunks, text))
        notes.extend(_verbose_notes(sources, images, device, request))
    return RunReport(text=text, images=image_results, warnings=notes)


def _collect_images(sources: list) -> list[EmbeddedImage]:
    """Number every embedded image across all inputs in one sequence."""
    images = []
    for source in sources:
        for image in source.images:
            images.append(EmbeddedImage(len(images), image.mime_type, image.data))
    return images


def _scope_notes(sources: list, chunks: list) -> list[str]:
    """Say what a collective score was taken over, when there is more than one file."""
    if len(sources) < 2:
        return []
    return [
        f"This is one collective answer for {len(sources)} files, pooled over "
        f"{len(chunks)} sections of text. A single badly written file can move "
        f"it; --verbose breaks the answer down per file."
    ]


def _per_file_notes(chunks: list, text: list) -> list[str]:
    """Break a collective score back down into the files behind it."""
    names = list(dict.fromkeys(chunk.source for chunk in chunks))
    if len(names) < 2:
        return []

    notes = []
    for result in text:
        if not result.chunk_scores:
            continue
        parts = []
        for name in names:
            scores = [
                score
                for chunk, score in zip(chunks, result.chunk_scores)
                if chunk.source == name
            ]
            if scores:
                parts.append(f"{Path(name).name} {mean_score(scores) * 100:.0f}%")
        notes.append(f"{result.name} per file: " + ", ".join(parts))
    return notes


def _agreement_notes(text: list, verbose: bool) -> list[str]:
    """Report whether the probability detectors actually agree, chunk by chunk.

    Two detectors are only comparable because they score the same chunks; a
    mean alone cannot distinguish "they disagree" from "one is misread".
    """
    comparable = [
        result
        for result in text
        if result.name != EDITLENS_NAME and len(result.chunk_scores) >= 2
    ]
    notes = []
    for index, first in enumerate(comparable):
        for second in comparable[index + 1 :]:
            comparison = compare_detectors(first.chunk_scores, second.chunk_scores)
            if comparison is None:
                continue
            notes.append(
                _agreement_sentence(first, second, comparison)
                if verbose
                else _plain_agreement_sentence(first, second, comparison)
            )
    return notes


def _plain_agreement_sentence(first, second, comparison) -> str:
    """Say whether the detectors back each other up, without the statistics."""
    disagreed = comparison.chunks - comparison.agreed
    if (first.score >= 0.5) != (second.score >= 0.5):
        return (
            f"{first.name} and {second.name} flatly contradict each other about "
            "this page: one reads it as AI-generated and the other as human. "
            "When they disagree this sharply, neither reading is trustworthy "
            "here. Treat the page as unresolved."
        )
    if disagreed > comparison.chunks / 3:
        return (
            f"{first.name} and {second.name} reach the same overall answer but "
            f"disagree about {disagreed} of the {comparison.chunks} sections "
            "they read, so that answer is shaky."
        )
    return (
        f"{first.name} and {second.name} independently reach the same answer, "
        "which is mild support for it."
    )


def _agreement_sentence(first, second, comparison) -> str:
    """State the comparison in counts, which stay meaningful when kappa cannot."""
    disagreed = comparison.chunks - comparison.agreed
    head = (
        f"{first.name} and {second.name} agree on {comparison.agreed} of "
        f"{comparison.chunks} shared chunks"
    )
    if comparison.degenerate:
        return (
            f"{head}: one of them gave every chunk the same call, so Cohen's "
            f"kappa is 0.00 by construction and says nothing here. "
            f"They contradict each other on {disagreed}."
        )
    return f"{head} (Cohen's kappa {comparison.kappa:+.2f})."


def _unavailable_notes(text: list, images: list) -> list[str]:
    """Say plainly that this is a partial report, and which detectors are missing."""
    missing = [result.name for result in text if result.score is None]
    if images and all(image.skipped_reason for image in images):
        missing.append("the image detector")
    if not missing:
        return []
    return [
        "Not every detector ran: "
        + ", ".join(missing)
        + " could not start. The results shown are unaffected by that."
    ]


def _verbose_notes(
    sources: list, images: list, device: str, request: RunRequest
) -> list[str]:
    """Report extraction counts without echoing any input content."""
    embedded = len(images)
    characters = sum(len(source.text) for source in sources)
    notes = [
        f"Device: {device}.",
        f"Read {len(sources)} file(s).",
        f"Extracted {characters} characters of readable text.",
        f"Found {embedded} embedded raster image(s).",
    ]
    if embedded and not request.include_images:
        notes.append("Embedded images were not analyzed; pass --images to include them.")
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="slop_detector.main",
        description="Analyze .webarchive or Markdown files for likely AI-generated content.",
    )
    parser.add_argument("--input", required=True, action="append", type=Path)
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    request = RunRequest(
        paths=[str(path) for path in args.input],
        include_images=args.images,
        verbose=args.verbose,
    )
    try:
        report = build_report(request)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(report.to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
