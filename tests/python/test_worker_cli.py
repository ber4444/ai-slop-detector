import json

import pytest

from slop_detector.contracts import DetectorResult, FileScore
from slop_detector.main import main, read_hf_token
from slop_detector.models import Chunk


def fake_source(name="note.md", text="Text."):
    return type("Source", (), {"name": name, "text": text})()


@pytest.fixture
def stub_worker(monkeypatch):
    """Replace input reading, device selection, and models with stubs."""
    monkeypatch.setattr("slop_detector.main.load_sources", lambda _: [fake_source()])
    monkeypatch.setattr(
        "slop_detector.main.chunk_sources",
        lambda sources: [Chunk(source=s.name, text=s.text) for s in sources],
    )
    monkeypatch.setattr("slop_detector.main.select_runtime_device", lambda: ("mps", []))
    monkeypatch.setattr("slop_detector.main.read_hf_token", lambda: "token")
    monkeypatch.setattr("slop_detector.main.TransformersLoader", lambda token: object())
    monkeypatch.setattr("slop_detector.main.run_text_detectors", lambda *args: [])
    return monkeypatch


def test_worker_emits_a_machine_readable_report(stub_worker, capsys, tmp_path):
    stub_worker.setattr(
        "slop_detector.main.run_text_detectors",
        lambda *args: [DetectorResult("Glyph", 0.4, "probably human", "1 chunk", [0.4])],
    )

    assert main(["--input", str(tmp_path / "note.md")]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["text"][0]["name"] == "Glyph"
    assert "images" not in report


def test_worker_always_prints_the_advisory_notes(stub_worker, capsys, tmp_path):
    main(["--input", str(tmp_path / "note.md")])

    warnings = json.loads(capsys.readouterr().out)["warnings"]
    assert any("guesses, not proof" in warning for warning in warnings)


def test_worker_reports_verbose_extraction_counts_only_when_asked(
    stub_worker, capsys, tmp_path
):
    stub_worker.setattr(
        "slop_detector.main.run_text_detectors",
        lambda *args: [DetectorResult("Glyph", 0.5, "unclear", "1 chunk", [0.5])],
    )

    main(["--input", str(tmp_path / "note.md"), "--verbose"])
    verbose = json.loads(capsys.readouterr().out)["warnings"]
    main(["--input", str(tmp_path / "note.md")])
    quiet = json.loads(capsys.readouterr().out)["warnings"]

    assert any("characters of readable text" in warning for warning in verbose)
    assert not any("characters of readable text" in warning for warning in quiet)


def test_worker_rejects_an_unreadable_input_with_exit_code_two(
    stub_worker, capsys, tmp_path
):
    def refuse(_):
        raise ValueError("Unsupported file type: page.html")

    stub_worker.setattr("slop_detector.main.load_sources", refuse)

    assert main(["--input", str(tmp_path / "page.html")]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Unsupported file type" in captured.err


def test_worker_reports_model_failures_with_exit_code_one(
    stub_worker, capsys, tmp_path
):
    def fail(*args):
        raise RuntimeError("ogmatrixllm/glyph-v1.1: gated repository")

    stub_worker.setattr("slop_detector.main.run_text_detectors", fail)

    assert main(["--input", str(tmp_path / "note.md")]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "glyph-v1.1" in captured.err


def test_missing_token_names_the_gitignored_token_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match=r"\.secrets/huggingface\.token"):
        read_hf_token()


def test_blank_token_file_is_rejected_without_echoing_its_contents(
    monkeypatch, tmp_path
):
    secrets = tmp_path / ".secrets"
    secrets.mkdir()
    (secrets / "huggingface.token").write_text("   \n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match=r"\.secrets/huggingface\.token"):
        read_hf_token()


def test_token_is_read_trimmed_from_the_secrets_directory(monkeypatch, tmp_path):
    secrets = tmp_path / ".secrets"
    secrets.mkdir()
    (secrets / "huggingface.token").write_text("hf_example\n")
    monkeypatch.chdir(tmp_path)

    assert read_hf_token() == "hf_example"


def test_partial_text_results_are_reported_with_a_named_gap(
    stub_worker, capsys, tmp_path
):
    stub_worker.setattr(
        "slop_detector.main.run_text_detectors",
        lambda *args: [
            DetectorResult("Glyph", 0.4, "probably human", "1 chunk", [0.4]),
            DetectorResult("EditLens", None, "not assessed", "gated"),
        ],
    )

    assert main(["--input", str(tmp_path / "note.md")]) == 0

    report = json.loads(capsys.readouterr().out)
    assert [result["score"] for result in report["text"]] == [0.4, None]
    assert any("Not every detector ran" in note for note in report["warnings"])
    assert any("EditLens" in note for note in report["warnings"])


def test_a_run_with_no_usable_detector_fails_instead_of_reporting_nothing(
    stub_worker, capsys, tmp_path
):
    stub_worker.setattr(
        "slop_detector.main.run_text_detectors",
        lambda *args: [
            DetectorResult("Glyph", None, "not assessed", "gated repository"),
            DetectorResult("EditLens", None, "not assessed", "gated"),
        ],
    )

    assert main(["--input", str(tmp_path / "note.md")]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No text detector could run" in captured.err
    assert "Glyph" in captured.err


def test_agreement_between_detectors_is_reported_as_kappa(
    stub_worker, capsys, tmp_path
):
    stub_worker.setattr(
        "slop_detector.main.run_text_detectors",
        lambda *args: [
            DetectorResult("Glyph", 0.5, "x", "4 chunks", [0.9, 0.9, 0.1, 0.1]),
            DetectorResult("Vanguard", 0.5, "x", "4 chunks", [0.1, 0.1, 0.9, 0.9]),
        ],
    )

    main(["--input", str(tmp_path / "note.md"), "--verbose"])

    warnings = json.loads(capsys.readouterr().out)["warnings"]
    agreement = next(note for note in warnings if "kappa" in note)
    assert "Glyph and Vanguard agree on 0 of 4 shared chunks" in agreement
    assert "-1.00" in agreement


def test_editlens_is_left_out_of_the_agreement_statistic(stub_worker, capsys, tmp_path):
    stub_worker.setattr(
        "slop_detector.main.run_text_detectors",
        lambda *args: [
            DetectorResult("Glyph", 0.5, "x", "2 chunks", [0.9, 0.1]),
            DetectorResult("EditLens", 0.5, "y", "2 chunks", [0.9, 0.1]),
        ],
    )

    main(["--input", str(tmp_path / "note.md")])

    warnings = json.loads(capsys.readouterr().out)["warnings"]
    assert not any("kappa" in note for note in warnings)


def test_agreement_note_refuses_to_read_meaning_into_a_degenerate_kappa(
    stub_worker, capsys, tmp_path
):
    stub_worker.setattr(
        "slop_detector.main.run_text_detectors",
        lambda *args: [
            DetectorResult("Glyph", 0.87, "x", "14 chunks", [0.9] * 13 + [0.04]),
            DetectorResult("Vanguard", 0.08, "y", "14 chunks", [0.05] * 14),
        ],
    )

    main(["--input", str(tmp_path / "note.md"), "--verbose"])

    warnings = json.loads(capsys.readouterr().out)["warnings"]
    note = next(note for note in warnings if "Glyph and Vanguard" in note)
    assert "agree on 1 of 14 shared chunks" in note
    assert "says nothing here" in note


def test_survey_counts_how_many_files_each_detector_flags(stub_worker, capsys):
    stub_worker.setattr(
        "slop_detector.main.load_sources",
        lambda _: [fake_source("a.md"), fake_source("b.md"), fake_source("c.md")],
    )
    stub_worker.setattr(
        "slop_detector.main.run_text_detectors",
        lambda *args: [
            DetectorResult(
                "Glyph",
                0.4,
                "x",
                "3 chunks",
                [0.2, 0.9, 0.95],
                [
                    FileScore("a.md", 0.2, 1),
                    FileScore("b.md", 0.9, 1),
                    FileScore("c.md", 0.95, 1),
                ],
            )
        ],
    )

    main(["--input", "a.md", "--input", "b.md", "--input", "c.md"])

    warnings = json.loads(capsys.readouterr().out)["warnings"]
    note = next(note for note in warnings if "Glyph flags" in note)
    assert "flags 2 of 3 files as AI-generated" in note
    assert "false-positive rate" in note


def test_a_single_file_gets_no_survey_note(stub_worker, capsys, tmp_path):
    stub_worker.setattr(
        "slop_detector.main.run_text_detectors",
        lambda *args: [
            DetectorResult("Glyph", 0.4, "x", "1 chunk", [0.4], [FileScore("a.md", 0.4, 1)])
        ],
    )

    main(["--input", str(tmp_path / "note.md")])

    warnings = json.loads(capsys.readouterr().out)["warnings"]
    assert not any("flags" in note for note in warnings)


def test_editlens_is_left_out_of_the_flag_count(stub_worker, capsys):
    stub_worker.setattr(
        "slop_detector.main.load_sources",
        lambda _: [fake_source("a.md"), fake_source("b.md")],
    )
    stub_worker.setattr(
        "slop_detector.main.run_text_detectors",
        lambda *args: [
            DetectorResult(
                "EditLens",
                0.9,
                "heavy",
                "2 chunks",
                [0.9, 0.9],
                [FileScore("a.md", 0.9, 1), FileScore("b.md", 0.9, 1)],
            )
        ],
    )

    main(["--input", "a.md", "--input", "b.md"])

    warnings = json.loads(capsys.readouterr().out)["warnings"]
    assert not any("flags" in note for note in warnings)
