import json

import pytest

from slop_detector.contracts import DetectorResult, ImageResult
from slop_detector.main import main, read_hf_token


def fake_content(text="Text.", images=(object(),)):
    return type("Content", (), {"text": text, "images": list(images)})()


@pytest.fixture
def stub_worker(monkeypatch):
    """Replace archive parsing, device selection, and models with stubs."""
    monkeypatch.setattr("slop_detector.main.parse_webarchive", lambda _: fake_content())
    monkeypatch.setattr("slop_detector.main.select_runtime_device", lambda: ("mps", []))
    monkeypatch.setattr("slop_detector.main.read_hf_token", lambda: "token")
    monkeypatch.setattr("slop_detector.main.TransformersLoader", lambda token: object())
    monkeypatch.setattr("slop_detector.main.run_text_detectors", lambda *args: [])
    return monkeypatch


def test_worker_emits_machine_readable_report_without_images(
    stub_worker, capsys, tmp_path
):
    assert main(["--archive", str(tmp_path / "page.webarchive")]) == 0

    assert json.loads(capsys.readouterr().out)["images"] == []


def test_worker_reports_individual_bad_images_without_discarding_text(
    stub_worker, capsys, tmp_path
):
    stub_worker.setattr("slop_detector.main.run_text_detectors", lambda *args: [])
    stub_worker.setattr(
        "slop_detector.main.run_image_detector",
        lambda *args: [ImageResult(0, None, None, [], "OSError: broken")],
    )

    assert main(["--archive", str(tmp_path / "page.webarchive"), "--images"]) == 0

    images = json.loads(capsys.readouterr().out)["images"]
    assert images == [
        {
            "index": 0,
            "score": None,
            "label": None,
            "metadata_flags": [],
            "skipped_reason": "OSError: broken",
        }
    ]


def test_worker_always_prints_the_advisory_notes(stub_worker, capsys, tmp_path):
    main(["--archive", str(tmp_path / "page.webarchive")])

    warnings = json.loads(capsys.readouterr().out)["warnings"]
    assert any("advisory" in warning for warning in warnings)


def test_worker_reports_verbose_extraction_counts_only_when_asked(
    stub_worker, capsys, tmp_path
):
    stub_worker.setattr(
        "slop_detector.main.run_text_detectors",
        lambda *args: [DetectorResult("Glyph", 0.5, "likely AI-generated", "1 chunk")],
    )

    main(["--archive", str(tmp_path / "page.webarchive"), "--verbose"])
    verbose = json.loads(capsys.readouterr().out)["warnings"]
    main(["--archive", str(tmp_path / "page.webarchive")])
    quiet = json.loads(capsys.readouterr().out)["warnings"]

    assert any("characters of readable text" in warning for warning in verbose)
    assert not any("characters of readable text" in warning for warning in quiet)


def test_worker_rejects_an_invalid_archive_with_exit_code_two(
    stub_worker, capsys, tmp_path
):
    def refuse(_):
        raise ValueError("Invalid webarchive: not a plist")

    stub_worker.setattr("slop_detector.main.parse_webarchive", refuse)

    assert main(["--archive", str(tmp_path / "page.webarchive")]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Invalid webarchive" in captured.err


def test_worker_reports_model_failures_with_exit_code_one(
    stub_worker, capsys, tmp_path
):
    def fail(*args):
        raise RuntimeError("ogmatrixllm/glyph-v1.1: gated repository")

    stub_worker.setattr("slop_detector.main.run_text_detectors", fail)

    assert main(["--archive", str(tmp_path / "page.webarchive")]) == 1

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
            DetectorResult("Glyph", 0.4, "likely AI-generated", "1 chunk"),
            DetectorResult("EditLens", None, "estimated AI-edit extent", "gated"),
        ],
    )

    assert main(["--archive", str(tmp_path / "page.webarchive")]) == 0

    report = json.loads(capsys.readouterr().out)
    assert [result["score"] for result in report["text"]] == [0.4, None]
    assert any("partial report" in note for note in report["warnings"])
    assert any("EditLens" in note for note in report["warnings"])


def test_a_run_with_no_usable_detector_fails_instead_of_reporting_nothing(
    stub_worker, capsys, tmp_path
):
    stub_worker.setattr(
        "slop_detector.main.run_text_detectors",
        lambda *args: [
            DetectorResult("Glyph", None, "likely AI-generated", "gated repository"),
            DetectorResult("EditLens", None, "estimated AI-edit extent", "gated"),
        ],
    )

    assert main(["--archive", str(tmp_path / "page.webarchive")]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No text detector could run" in captured.err
    assert "Glyph" in captured.err


def test_a_complete_run_says_nothing_about_partial_results(
    stub_worker, capsys, tmp_path
):
    stub_worker.setattr(
        "slop_detector.main.run_text_detectors",
        lambda *args: [DetectorResult("Glyph", 0.4, "likely AI-generated", "1 chunk")],
    )

    main(["--archive", str(tmp_path / "page.webarchive")])

    warnings = json.loads(capsys.readouterr().out)["warnings"]
    assert not any("partial report" in note for note in warnings)


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

    main(["--archive", str(tmp_path / "page.webarchive")])

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

    main(["--archive", str(tmp_path / "page.webarchive")])

    warnings = json.loads(capsys.readouterr().out)["warnings"]
    assert not any("kappa" in note for note in warnings)


def test_no_agreement_note_when_only_one_detector_scored(stub_worker, capsys, tmp_path):
    stub_worker.setattr(
        "slop_detector.main.run_text_detectors",
        lambda *args: [
            DetectorResult("Glyph", 0.5, "x", "2 chunks", [0.9, 0.1]),
            DetectorResult("Vanguard", None, "not assessed", "gated"),
        ],
    )

    main(["--archive", str(tmp_path / "page.webarchive")])

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

    main(["--archive", str(tmp_path / "page.webarchive")])

    warnings = json.loads(capsys.readouterr().out)["warnings"]
    note = next(note for note in warnings if "Glyph and Vanguard" in note)
    assert "agree on 1 of 14 shared chunks" in note
    assert "says nothing here" in note
    assert "contradict each other on 13" in note
