from slop_detector.contracts import DetectorResult, ImageResult, RunReport


def test_report_json_round_trip_preserves_separate_text_image_and_metadata_results():
    report = RunReport(
        text=[DetectorResult("Glyph", 0.75, "likely AI-generated", "2 chunks")],
        images=[
            ImageResult(
                0,
                0.9,
                "artificial",
                ["Software: Stable Diffusion"],
                None,
            )
        ],
        warnings=["Detector scores are advisory."],
    )

    assert RunReport.from_json(report.to_json()) == report
