"""Black-box tests that run the Kotlin CLI against a fake Python worker."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[2]
SCRIPT = "slop-detector.main.kts"


def detector(name, score, label, detail="1 chunk", chunk_scores=None, file_scores=None):
    return {
        "name": name,
        "score": score,
        "label": label,
        "detail": detail,
        "chunk_scores": chunk_scores or [],
        "file_scores": file_scores or [],
    }


SINGLE_REPORT = {
    "text": [
        detector("Glyph", 0.75, "probably AI-generated", chunk_scores=[0.75]),
        detector("EditLens", 0.2, "little sign of AI editing", chunk_scores=[0.2]),
    ],
    "warnings": ["These are guesses, not proof."],
}

pytestmark = pytest.mark.skipif(
    shutil.which("kotlin") is None, reason="the Kotlin compiler is not installed"
)


def fake_python(tmp_path, report=SINGLE_REPORT, exit_code=0, stderr=""):
    """Create a stand-in worker that records its arguments and replies."""
    received = tmp_path / "received-args"
    executable = tmp_path / "python3"
    executable.write_text(
        "#!/bin/sh\n"
        f'printf "%s" "$*" > {received}\n'
        f"printf '%s' {json.dumps(json.dumps(report))}\n"
        f'printf "%s" {json.dumps(stderr)} >&2\n'
        f"exit {exit_code}\n"
    )
    executable.chmod(0o755)
    return executable, received


def run_cli(tmp_path, *arguments, report=SINGLE_REPORT, exit_code=0, stderr=""):
    executable, received = fake_python(
        tmp_path, report=report, exit_code=exit_code, stderr=stderr
    )
    result = subprocess.run(
        ["kotlin", SCRIPT, *arguments],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "SLOP_DETECTOR_PYTHON": str(executable),
            "SLOP_DETECTOR_SKIP_BOOTSTRAP": "1",
        },
    )
    return result, received


def test_kotlin_cli_renders_a_single_file_verdict(tmp_path):
    note = tmp_path / "note.md"
    note.write_text("# Title\n\nProse.\n")

    result, received = run_cli(tmp_path, str(note))

    assert result.returncode == 0, result.stderr
    assert f"Read: {note}" in result.stdout
    assert "Glyph: probably AI-generated" in result.stdout
    assert "These are guesses, not proof." in result.stdout
    arguments = received.read_text()
    assert "--input" in arguments


def test_kotlin_cli_shows_percentages_only_under_verbose(tmp_path):
    note = tmp_path / "note.md"
    note.write_text("prose")

    quiet, _ = run_cli(tmp_path, str(note))
    loud, _ = run_cli(tmp_path, str(note), "--verbose")

    assert "75.0%" not in quiet.stdout
    assert "Glyph: 75.0% — probably AI-generated" in loud.stdout


def test_kotlin_cli_propagates_the_worker_exit_code_and_message(tmp_path):
    note = tmp_path / "note.md"
    note.write_text("prose")

    result, _ = run_cli(
        tmp_path,
        str(note),
        exit_code=2,
        stderr="Unsupported file type: page.html\n",
    )

    assert result.returncode == 2
    assert "Unsupported file type" in result.stderr
    assert result.stdout.strip() == ""


def test_kotlin_cli_rejects_an_unsupported_file_before_starting_the_worker(tmp_path):
    page = tmp_path / "page.webarchive"
    page.write_bytes(b"no longer supported")

    result, received = run_cli(tmp_path, str(page))

    assert result.returncode == 2
    assert "Unsupported file type" in result.stderr
    assert not received.exists()


def test_kotlin_cli_prints_usage_without_bootstrapping(tmp_path):
    result, received = run_cli(tmp_path, "--help")

    assert result.returncode == 0
    assert "Usage: kotlin slop-detector.main.kts" in result.stdout
    assert not received.exists()


def test_kotlin_cli_renders_an_unavailable_detector_with_its_reason(tmp_path):
    note = tmp_path / "note.md"
    note.write_text("prose")
    report = {
        "text": [
            SINGLE_REPORT["text"][0],
            detector(
                "EditLens",
                None,
                "not assessed",
                detail=(
                    "pangram/editlens_Llama-3.2-3B: GatedRepoError: 403\n"
                    "Accept the access conditions for both model pages."
                ),
            ),
        ],
        "warnings": ["Not every detector ran: EditLens could not start."],
    }

    result, _ = run_cli(tmp_path, str(note), report=report)

    assert result.returncode == 0
    assert "EditLens: could not run" in result.stdout
    assert "GatedRepoError: 403" in result.stdout
    assert "Accept the access conditions for both model pages." in result.stdout


# --- corpus survey ---------------------------------------------------------

def survey_report(flagged, total):
    """A Glyph result over `total` files, `flagged` of them scoring high."""
    files = [
        {"name": f"doc-{i}.md", "score": 0.9 if i < flagged else 0.05, "chunks": 1}
        for i in range(total)
    ]
    scores = [f["score"] for f in files]
    return {
        "text": [
            detector("Glyph", sum(scores) / len(scores), "x", f"{total} chunks", scores, files)
        ],
        "warnings": [
            f"Glyph flags {flagged} of {total} files as AI-generated "
            f"({flagged / total * 100:.1f}%). If these files are all "
            f"human-written, that is its false-positive rate on this kind of writing."
        ],
    }


def test_kotlin_cli_surveys_a_directory_with_counts_and_a_histogram(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(8):
        (corpus / f"doc-{i}.md").write_text("prose")

    executable, received = fake_python(tmp_path, report=survey_report(flagged=2, total=8))
    result = subprocess.run(
        ["kotlin", SCRIPT, str(corpus)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "SLOP_DETECTOR_PYTHON": str(executable),
            "SLOP_DETECTOR_SKIP_BOOTSTRAP": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "Surveyed 8 files, each scored on its own" in result.stdout
    assert "Glyph flags 2 of 8 files as likely AI-generated" in result.stdout
    assert "files it flags:" in result.stdout
    assert "█" in result.stdout  # a histogram bar
    assert "aggregate over all text pooled" in result.stdout
    assert received.read_text().count("--input") == 8


def test_kotlin_cli_forwards_every_markdown_file_a_glob_matches(tmp_path):
    for name in ("alpha.md", "beta.md", "gamma.md"):
        (tmp_path / name).write_text("prose")
    (tmp_path / "ignored.txt").write_text("not markdown")

    executable, received = fake_python(tmp_path, report=survey_report(flagged=0, total=3))
    result = subprocess.run(
        ["kotlin", SCRIPT, str(tmp_path / "*.md")],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "SLOP_DETECTOR_PYTHON": str(executable),
            "SLOP_DETECTOR_SKIP_BOOTSTRAP": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    arguments = received.read_text()
    assert arguments.count("--input") == 3
    assert "ignored.txt" not in arguments


def test_kotlin_cli_reports_a_glob_that_matches_nothing(tmp_path):
    executable, received = fake_python(tmp_path)
    result = subprocess.run(
        ["kotlin", SCRIPT, str(tmp_path / "*.md")],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "SLOP_DETECTOR_PYTHON": str(executable),
            "SLOP_DETECTOR_SKIP_BOOTSTRAP": "1",
        },
    )

    assert result.returncode == 2
    assert "No matching files" in result.stderr
    assert not received.exists()
