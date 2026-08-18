"""Black-box tests that run the Kotlin CLI against a fake Python worker."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[2]
SCRIPT = "slop-detector.main.kts"

REPORT = {
    "text": [
        {
            "name": "Glyph",
            "score": 0.75,
            "label": "likely AI-generated",
            "detail": "1 chunks",
        },
        {
            "name": "EditLens",
            "score": 0.2,
            "label": "estimated AI-edit extent",
            "detail": "1 chunks",
        },
    ],
    "images": [
        {
            "index": 0,
            "score": 0.9,
            "label": "likely artificial",
            "metadata_flags": ["Software: Stable Diffusion"],
            "skipped_reason": None,
        },
        {
            "index": 1,
            "score": None,
            "label": None,
            "metadata_flags": [],
            "skipped_reason": "OSError: broken",
        },
    ],
    "warnings": ["Detector scores are advisory."],
}

pytestmark = pytest.mark.skipif(
    shutil.which("kotlin") is None, reason="the Kotlin compiler is not installed"
)


def fake_python(tmp_path, report=REPORT, exit_code=0, stderr=""):
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


def run_cli(tmp_path, *arguments, report=REPORT, exit_code=0, stderr=""):
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


def test_kotlin_cli_renders_worker_report_and_forwards_images_flag(tmp_path):
    result, received = run_cli(tmp_path, "page.webarchive", "--images")

    assert result.returncode == 0, result.stderr
    assert "Glyph: likely AI-generated" in result.stdout
    assert "EditLens: estimated AI-edit extent" in result.stdout
    assert "Detector scores are advisory." in result.stdout
    arguments = received.read_text()
    assert "--images" in arguments
    assert "--input" in arguments


def test_kotlin_cli_keeps_image_scores_metadata_and_skips_in_separate_sections(
    tmp_path,
):
    result, _ = run_cli(tmp_path, "page.webarchive", "--images")

    assert "What the image detector thinks" in result.stdout
    assert "Image 0: likely artificial" in result.stdout
    assert "Image 1: skipped (OSError: broken)" in result.stdout
    assert "What the image files say about themselves" in result.stdout
    assert "Software: Stable Diffusion" in result.stdout


def test_kotlin_cli_omits_image_sections_when_no_images_were_analyzed(tmp_path):
    result, received = run_cli(tmp_path, "page.webarchive", report={**REPORT, "images": []})

    assert "What the image detector thinks" not in result.stdout
    assert "What the image files say about themselves" not in result.stdout
    assert "--images" not in received.read_text()


def test_kotlin_cli_propagates_the_worker_exit_code_and_message(tmp_path):
    result, _ = run_cli(
        tmp_path,
        "page.webarchive",
        exit_code=2,
        stderr="Invalid webarchive: not a plist\n",
    )

    assert result.returncode == 2
    assert "Invalid webarchive" in result.stderr
    assert result.stdout.strip() == ""


def test_kotlin_cli_rejects_a_non_webarchive_argument_before_starting_the_worker(
    tmp_path,
):
    result, received = run_cli(tmp_path, "page.html")

    assert result.returncode == 2
    assert "webarchive" in result.stderr
    assert not received.exists()


def test_kotlin_cli_prints_usage_without_bootstrapping(tmp_path):
    result, received = run_cli(tmp_path, "--help")

    assert result.returncode == 0
    assert "Usage: kotlin slop-detector.main.kts" in result.stdout
    assert not received.exists()


def test_kotlin_cli_renders_an_unavailable_detector_with_its_reason(tmp_path):
    report = {
        **REPORT,
        "text": [
            REPORT["text"][0],
            {
                "name": "EditLens",
                "score": None,
                "label": "estimated AI-edit extent",
                "detail": (
                    "pangram/editlens_Llama-3.2-3B: GatedRepoError: 403\n"
                    "Accept the access conditions for both model pages."
                ),
            },
        ],
        "warnings": ["This is a partial report: EditLens could not run."],
    }

    result, _ = run_cli(tmp_path, "page.webarchive", report=report)

    assert result.returncode == 0
    assert "Glyph: likely AI-generated" in result.stdout
    assert "EditLens: could not run" in result.stdout
    assert "GatedRepoError: 403" in result.stdout
    assert "Accept the access conditions for both model pages." in result.stdout
    assert "This is a partial report: EditLens could not run." in result.stdout


def test_kotlin_cli_forwards_every_markdown_file_a_glob_matches(tmp_path):
    for name in ("alpha.md", "beta.md", "gamma.md"):
        (tmp_path / name).write_text("# Title\n\nSome prose.\n")
    (tmp_path / "ignored.txt").write_text("not markdown")

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

    assert result.returncode == 0, result.stderr
    arguments = received.read_text()
    assert arguments.count("--input") == 3
    assert "alpha.md" in arguments and "gamma.md" in arguments
    assert "ignored.txt" not in arguments
    assert "3 files, scored together as one answer" in result.stdout


def test_kotlin_cli_walks_a_directory_for_supported_files(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "one.md").write_text("prose")
    (tmp_path / "docs" / "nested").mkdir()
    (tmp_path / "docs" / "nested" / "two.markdown").write_text("prose")
    (tmp_path / "docs" / "skip.png").write_bytes(b"\x89PNG")

    executable, received = fake_python(tmp_path)
    result = subprocess.run(
        ["kotlin", SCRIPT, str(tmp_path / "docs")],
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
    assert arguments.count("--input") == 2
    assert "two.markdown" in arguments
    assert "skip.png" not in arguments


def test_kotlin_cli_accepts_a_single_markdown_file(tmp_path):
    note = tmp_path / "note.md"
    note.write_text("prose")

    executable, received = fake_python(tmp_path)
    result = subprocess.run(
        ["kotlin", SCRIPT, str(note)],
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
    assert f"Read: {note}" in result.stdout


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
