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
