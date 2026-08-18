from slop_detector.contracts import DetectorResult, FileScore, RunReport


def test_report_json_round_trip_preserves_per_file_scores():
    report = RunReport(
        text=[
            DetectorResult(
                "Glyph",
                0.42,
                "unclear — could be either",
                "5 chunks",
                chunk_scores=[0.1, 0.9, 0.3, 0.5, 0.3],
                file_scores=[
                    FileScore("a.md", 0.2, 3),
                    FileScore("b.md", 0.9, 2),
                ],
            )
        ],
        warnings=["These are guesses, not proof."],
    )

    assert RunReport.from_json(report.to_json()) == report


def test_report_json_round_trip_survives_a_detector_that_could_not_run():
    report = RunReport(
        text=[DetectorResult("EditLens", None, "not assessed", "gated")],
        warnings=[],
    )

    assert RunReport.from_json(report.to_json()) == report
