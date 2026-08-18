import pytest

from slop_detector.models import (
    chunk_text,
    editlens_score,
    mean_score,
    run_image_detector,
    ai_probability,
    cohens_kappa,
    compare_detectors,
    verdict,
    resolve_ai_label_index,
    run_text_detectors,
    select_device,
)
from slop_detector.models import TEXT_MODELS
from slop_detector.webarchive import EmbeddedImage


def test_chunk_text_keeps_sentences_and_honors_the_limit():
    assert chunk_text("One. Two. Three.", max_characters=10) == ["One. Two.", "Three."]


def test_chunk_text_splits_a_single_overlong_sentence_as_a_last_resort():
    chunks = chunk_text("abcdefghij klmnopqrst", max_characters=10)

    assert chunks == ["abcdefghij", "klmnopqrst"]
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_chunk_text_drops_empty_chunks():
    assert chunk_text("   \n\n  ") == []


def test_editlens_score_is_normalized_expected_bucket_value():
    assert editlens_score([0.0, 0.0, 0.0]) == 0.5


def test_editlens_score_rejects_fewer_than_two_logits():
    with pytest.raises(ValueError):
        editlens_score([1.0])


def test_mean_score_of_no_scores_is_zero():
    assert mean_score([]) == 0.0
    assert mean_score([0.25, 0.75]) == 0.5


def test_select_device_prefers_mps_then_cpu():
    def torch_with_mps(available: bool) -> object:
        mps = type("M", (), {"is_available": staticmethod(lambda: available)})()
        return type("Torch", (), {"backends": type("B", (), {"mps": mps})()})()

    assert select_device(torch_with_mps(True)) == "mps"
    assert select_device(torch_with_mps(False)) == "cpu"


class FakeLoader:
    """Records load order so tests can assert models run one at a time."""

    def __init__(self, image_score=0.9, image_error=None):
        self.loaded = []
        self.chunks_seen = {}
        self.image_score = image_score
        self.image_error = image_error

    def load_text(self, model, device):
        self.loaded.append(model.model_id)
        seen = self.chunks_seen.setdefault(model.name, [])
        return lambda chunk: (seen.append(chunk), {"ai": 0.8})[1]

    def load_editlens(self, base_id, adapter_id, device):
        self.loaded.append(adapter_id)
        seen = self.chunks_seen.setdefault("EditLens", [])
        return lambda chunk: (seen.append(chunk), [0.0, 0.0, 0.0])[1]

    def load_image(self, model_id, device):
        self.loaded.append(model_id)

        def predict(data):
            if self.image_error is not None:
                raise self.image_error
            return self.image_score

        return predict


def test_text_detectors_run_sequentially_and_keep_their_distinct_meaning():
    loader = FakeLoader()

    results = run_text_detectors("A short sentence.", loader, "mps")

    assert [result.name for result in results] == ["Glyph", "Vanguard", "EditLens"]
    assert [result.score for result in results] == [0.8, 0.8, 0.5]
    assert loader.loaded == [
        "ogmatrixllm/glyph-v1.1",
        "ShantanuT01/vanguard-ai-text-detector",
        "pangram/editlens_Llama-3.2-3B",
    ]
    assert results[2].label == "estimated AI-edit extent"
    assert all(result.detail == "1 chunk" for result in results)


def test_text_detectors_report_empty_text_without_calling_a_model():
    loader = FakeLoader()

    results = run_text_detectors("   ", loader, "cpu")

    assert loader.loaded == []
    assert [result.detail for result in results] == ["no readable text"] * 3
    assert [result.score for result in results] == [0.0, 0.0, 0.0]


def test_image_detector_keeps_model_scores_and_metadata_flags_separate(png_bytes):
    loader = FakeLoader(image_score=0.9)

    results = run_image_detector(
        [EmbeddedImage(0, "image/png", png_bytes(Software="Stable Diffusion"))],
        loader,
        "mps",
    )

    assert loader.loaded == ["Organika/sdxl-detector"]
    assert results[0].score == 0.9
    assert results[0].label == "likely artificial"
    assert results[0].metadata_flags == ["Software: Stable Diffusion"]
    assert results[0].skipped_reason is None


def test_image_detector_skips_an_unreadable_image_without_failing_the_run(png_bytes):
    loader = FakeLoader(image_error=OSError("cannot identify image file"))

    results = run_image_detector(
        [
            EmbeddedImage(0, "image/png", b"not an image"),
            EmbeddedImage(1, "image/png", png_bytes()),
        ],
        loader,
        "cpu",
    )

    assert [result.index for result in results] == [0, 1]
    assert all(result.score is None for result in results)
    assert all(result.skipped_reason for result in results)
    assert loader.loaded == ["Organika/sdxl-detector"]


def test_image_detector_without_images_loads_nothing():
    loader = FakeLoader()

    assert run_image_detector([], loader, "mps") == []
    assert loader.loaded == []


def test_ai_label_index_comes_from_each_model_id2label():
    assert resolve_ai_label_index({0: "human", 1: "ai"}, "glyph") == 1
    assert resolve_ai_label_index({0: "artificial", 1: "human"}, "sdxl") == 0
    assert resolve_ai_label_index({0: "LABEL_0", 1: "machine-generated"}, "x") == 1
    assert resolve_ai_label_index({0: "human-written", 1: "LABEL_1"}, "x") == 1


def test_ai_label_index_refuses_to_guess_an_unknown_label_set():
    with pytest.raises(RuntimeError, match="vanguard"):
        resolve_ai_label_index({0: "LABEL_0", 1: "LABEL_1"}, "vanguard")


def test_ai_probability_reads_a_single_sigmoid_head_as_p_of_ai():
    assert ai_probability([0.0], None) == 0.5
    assert ai_probability([2.0], None) > 0.88
    assert ai_probability([-2.0], None) < 0.12


def test_ai_probability_softmaxes_a_multi_class_head_at_the_ai_index():
    assert ai_probability([0.0, 0.0], 1) == 0.5
    assert ai_probability([0.0, 10.0], 1) > 0.99
    assert ai_probability([0.0, 10.0], 0) < 0.01


def test_ai_probability_refuses_a_multi_class_head_without_an_index():
    with pytest.raises(ValueError):
        ai_probability([0.0, 0.0], None)


def test_glyph_is_pinned_to_its_documented_label_window_and_tokenizer():
    glyph = next(model for model in TEXT_MODELS if model.name == "Glyph")

    assert glyph.ai_index == 1
    assert glyph.fast_tokenizer is False
    assert glyph.max_characters <= 2048


def test_every_detector_scores_one_shared_partition():
    loader = FakeLoader()
    text = "Sentence number one. " * 400

    results = run_text_detectors(text, loader, "cpu")

    partitions = list(loader.chunks_seen.values())
    assert len(partitions) == 3
    assert all(partition == partitions[0] for partition in partitions)
    assert len(partitions[0]) > 1
    counts = {int(result.detail.split()[0]) for result in results}
    assert len(counts) == 1, "every detector must score the same partition"


class BrokenLoader(FakeLoader):
    """Fails the detectors named in `broken`, succeeds for the rest."""

    def __init__(self, *broken, image_load_error=None):
        super().__init__()
        self.broken = set(broken)
        self.image_load_error = image_load_error

    def load_text(self, model, device):
        if model.name in self.broken:
            raise RuntimeError(f"{model.model_id}: gated repository")
        return super().load_text(model, device)

    def load_editlens(self, base_id, adapter_id, device):
        if "EditLens" in self.broken:
            raise RuntimeError(f"{adapter_id}: gated repository\nAccept the terms.")
        return super().load_editlens(base_id, adapter_id, device)

    def load_image(self, model_id, device):
        if self.image_load_error is not None:
            raise self.image_load_error
        return super().load_image(model_id, device)


def test_a_gated_detector_does_not_discard_the_detectors_that_ran():
    results = run_text_detectors("A sentence.", BrokenLoader("EditLens"), "mps")

    scored = {result.name: result.score for result in results}
    assert scored["Glyph"] == 0.8
    assert scored["Vanguard"] == 0.8
    assert scored["EditLens"] is None


def test_an_unavailable_detector_keeps_the_guidance_its_error_carried():
    results = run_text_detectors("A sentence.", BrokenLoader("EditLens"), "mps")

    editlens = next(result for result in results if result.name == "EditLens")
    assert "gated repository" in editlens.detail
    assert "Accept the terms." in editlens.detail
    assert editlens.label == "not assessed"


def test_every_detector_can_fail_independently():
    results = run_text_detectors(
        "A sentence.", BrokenLoader("Glyph", "Vanguard", "EditLens"), "cpu"
    )

    assert [result.score for result in results] == [None, None, None]


def test_an_unloadable_image_model_skips_images_but_keeps_their_metadata(png_bytes):
    loader = BrokenLoader(image_load_error=RuntimeError("Organika: gated repository"))

    results = run_image_detector(
        [EmbeddedImage(0, "image/png", png_bytes(Software="Stable Diffusion"))],
        loader,
        "mps",
    )

    assert results[0].score is None
    assert "gated repository" in results[0].skipped_reason
    assert results[0].metadata_flags == ["Software: Stable Diffusion"]


def test_verbose_exposes_the_spread_the_mean_hides():
    class LabellingLoader(FakeLoader):
        def load_text(self, model, device):
            super().load_text(model, device)
            return lambda chunk: {"ai": 0.8, "label": "LABEL_1"}

    quiet = run_text_detectors("A sentence.", LabellingLoader(), "cpu")
    loud = run_text_detectors("A sentence.", LabellingLoader(), "cpu", verbose=True)

    assert quiet[0].detail == "1 chunk"
    assert "per-chunk 0.80-0.80" in loud[0].detail
    assert "1/1 over 0.50" in loud[0].detail


def test_verdict_states_the_plain_reading_on_each_side_of_the_threshold():
    assert verdict(0.216) == "likely human"
    assert verdict(0.371) == "likely human"
    assert verdict(0.0) == "likely human"
    assert verdict(0.9) == "likely AI-generated"
    assert verdict(1.0) == "likely AI-generated"


def test_verdict_declines_to_commit_near_the_decision_threshold():
    assert verdict(0.45) == "too close to call"
    assert verdict(0.55) == "too close to call"
    assert verdict(0.5) == "too close to call"


def test_verdict_uses_the_image_models_own_vocabulary():
    assert verdict(0.91, "artificial") == "likely artificial"
    assert verdict(0.05, "artificial") == "likely human"


def test_a_scored_detector_is_labelled_with_its_verdict_not_a_fixed_string():
    class HumanishLoader(FakeLoader):
        def load_text(self, model, device):
            super().load_text(model, device)
            return lambda chunk: {"ai": 0.216}

    results = run_text_detectors("A sentence.", HumanishLoader(), "cpu")

    assert results[0].label == "likely human"
    assert results[1].label == "likely human"


def test_detectors_carry_their_per_chunk_scores_for_comparison():
    results = run_text_detectors("One. Two. Three.", FakeLoader(), "cpu")

    assert results[0].chunk_scores == [0.8]
    assert results[1].chunk_scores == [0.8]
    assert results[2].chunk_scores == [0.5]


def test_an_unavailable_detector_carries_no_chunk_scores():
    results = run_text_detectors("A sentence.", BrokenLoader("EditLens"), "cpu")

    assert results[2].chunk_scores == []


def test_kappa_separates_agreement_from_chance_and_from_contradiction():
    agree = [0.9, 0.9, 0.1, 0.1]
    assert cohens_kappa(agree, [0.8, 0.7, 0.2, 0.3]) == 1.0
    assert cohens_kappa(agree, [0.1, 0.1, 0.9, 0.9]) == -1.0


def test_kappa_is_unavailable_when_the_chunks_are_not_shared():
    assert cohens_kappa([0.9, 0.1], [0.9]) is None
    assert cohens_kappa([0.9], [0.1]) is None


def test_kappa_of_a_detector_that_never_changes_its_mind_is_not_inflated():
    # Both call every chunk AI: they never disagree, but nothing is learned.
    assert cohens_kappa([0.9, 0.9, 0.9], [0.8, 0.8, 0.8]) == 1.0
    # One always says AI, the other is split: agreement is pure chance.
    assert cohens_kappa([0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.1, 0.1]) == 0.0


def test_comparison_flags_a_detector_that_never_changed_its_call():
    # Glyph calls 13 of 14 chunks AI, Vanguard calls none: kappa is 0 by
    # construction, so the counts have to carry the meaning.
    glyph = [0.9] * 13 + [0.04]
    vanguard = [0.05] * 14

    comparison = compare_detectors(glyph, vanguard)

    assert comparison.chunks == 14
    assert comparison.agreed == 1
    assert comparison.kappa == 0.0
    assert comparison.degenerate is True


def test_comparison_reports_a_real_kappa_when_both_detectors_vary():
    comparison = compare_detectors([0.9, 0.9, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1])

    assert comparison.degenerate is False
    assert comparison.agreed == 3
    assert 0.0 < comparison.kappa < 1.0
