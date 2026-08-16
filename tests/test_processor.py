import numpy as np
import pytest

from src.audio.processor import SignalProcessor


def synthetic_three_knocks(sample_rate: int = 16000) -> np.ndarray:
    waveform = np.zeros(sample_rate * 2, dtype=np.float32)
    waveform[[4000, 12000, 22000]] = 1.0
    return waveform


def test_legacy_processor_counts_and_extracts_deterministically() -> None:
    processor = SignalProcessor(model="legacy")
    waveform = synthetic_three_knocks(processor.sample_rate)

    first = processor.extract_features(waveform)
    second = processor.extract_features(waveform)

    assert processor.count_taps(waveform) == 3
    assert first.shape == (30,)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "waveform",
    [np.array([], dtype=np.float32), np.array([0.0, np.nan], dtype=np.float32)],
)
def test_processor_rejects_invalid_waveforms(waveform: np.ndarray) -> None:
    with pytest.raises(ValueError):
        SignalProcessor(model="legacy").extract_features(waveform)


def test_processor_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="model must be"):
        SignalProcessor(model="unknown")


def test_prepare_is_no_op_for_legacy_processor() -> None:
    processor = SignalProcessor(model="legacy")

    assert processor.prepare() is None


def test_prepare_eagerly_loads_and_caches_pann(monkeypatch) -> None:
    processor = SignalProcessor(model="pann")
    load_calls: list[str] = []

    def load_pann(checkpoint_path: str) -> None:
        load_calls.append(checkpoint_path)
        processor._pann_model = object()
        processor.checkpoint_sha256 = "test-sha256"

    monkeypatch.setattr(processor, "_load_pann", load_pann)

    assert processor.prepare() == "test-sha256"
    assert processor.prepare() == "test-sha256"
    assert load_calls == [str(processor.checkpoint_path)]
