import numpy as np
import pytest

from src.audio.processor import SignalProcessor


def synthetic_three_knocks(sample_rate: int = 16000) -> np.ndarray:
    waveform = np.zeros(sample_rate * 2, dtype=np.float32)
    burst_samples = round(0.025 * sample_rate)
    for start_seconds in (0.25, 0.75, 1.375):
        start = round(start_seconds * sample_rate)
        waveform[start : start + burst_samples] = 0.1
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


@pytest.mark.parametrize("model", ["legacy", "pann"])
def test_threshold_counter_counts_deliberate_knock_pattern(model: str) -> None:
    processor = SignalProcessor(model=model)
    waveform = synthetic_three_knocks(processor.sample_rate)

    assert processor.count_taps(waveform, energy_threshold=0.015) == 3


def test_threshold_counter_merges_ringing_into_one_knock() -> None:
    processor = SignalProcessor(model="legacy")
    waveform = np.zeros(processor.sample_rate, dtype=np.float32)
    for start_seconds, amplitude in ((0.20, 0.10), (0.25, 0.07), (0.30, 0.04)):
        start = round(start_seconds * processor.sample_rate)
        duration = round(0.025 * processor.sample_rate)
        waveform[start : start + duration] = amplitude

    assert processor.count_taps(waveform, energy_threshold=0.015) == 1


def test_threshold_counter_ignores_subthreshold_sound() -> None:
    processor = SignalProcessor(model="legacy")
    waveform = np.zeros(processor.sample_rate, dtype=np.float32)
    start = round(0.20 * processor.sample_rate)
    waveform[start : start + round(0.10 * processor.sample_rate)] = 0.014

    assert processor.count_taps(waveform, energy_threshold=0.015) == 0


def test_threshold_counter_requires_quiet_release() -> None:
    processor = SignalProcessor(model="legacy")
    waveform = np.zeros(processor.sample_rate, dtype=np.float32)
    for start_seconds in (0.20, 0.29, 0.55):
        start = round(start_seconds * processor.sample_rate)
        waveform[start : start + round(0.025 * processor.sample_rate)] = 0.1
    bridge_start = round(0.225 * processor.sample_rate)
    bridge_end = round(0.29 * processor.sample_rate)
    waveform[bridge_start:bridge_end] = 0.012

    assert processor.count_taps(waveform, energy_threshold=0.015) == 2


def test_threshold_counter_rejects_invalid_threshold() -> None:
    processor = SignalProcessor(model="legacy")
    waveform = np.zeros(processor.sample_rate, dtype=np.float32)

    with pytest.raises(ValueError, match="energy_threshold"):
        processor.count_taps(waveform, energy_threshold=-0.1)


def test_zero_threshold_ignores_silence_and_rearms() -> None:
    processor = SignalProcessor(model="legacy")
    waveform = np.zeros(processor.sample_rate, dtype=np.float32)
    for start_seconds in (0.20, 0.50):
        start = round(start_seconds * processor.sample_rate)
        waveform[start : start + round(0.025 * processor.sample_rate)] = 0.1

    assert processor.count_taps(waveform, energy_threshold=0.0) == 2


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
