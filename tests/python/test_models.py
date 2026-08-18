import pytest

from slop_detector.models import (
    chunk_text,
    editlens_score,
    mean_score,
    run_image_detector,
    ai_probability,
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
        self.chunk_limits = []
        self.image_score = image_score
        self.image_error = image_error

    def load_text(self, model, device):
        self.loaded.append(model.model_id)
        self.chunk_limits.append(model.max_characters)
        return lambda chunk: {"ai": 0.8}

    def load_editlens(self, base_id, adapter_id, device):
        self.loaded.append(adapter_id)
        return lambda chunk: [0.0, 0.0, 0.0]

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
    assert results[0].label == "artificial"
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


def test_each_detector_is_chunked_to_its_own_token_window():
    loader = FakeLoader()
    text = "Sentence number one. " * 400

    results = run_text_detectors(text, loader, "cpu")

    assert loader.chunk_limits == [1600, 6000]
    glyph, vanguard = results[0], results[1]
    assert int(glyph.detail.split()[0]) > int(vanguard.detail.split()[0])
