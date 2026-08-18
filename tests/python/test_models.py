import pytest

from slop_detector.models import chunk_text, editlens_score, mean_score, select_device


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
