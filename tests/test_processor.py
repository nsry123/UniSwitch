import numpy as np
import pytest

from src.audio.processor import (
    BEATS_CHECKPOINT_PATH,
    BEATS_EMBEDDING_DIM,
    BEATS_FEATURE_WINDOW_SECONDS,
    BEATS_PRE_ONSET_SECONDS,
    SignalProcessor,
)


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


@pytest.mark.parametrize("model", ["beats", "legacy", "pann"])
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


def test_beats_prepares_fixed_first_onset_window() -> None:
    processor = SignalProcessor(model="beats")
    waveform = np.zeros(14000, dtype=np.float32)
    waveform[5000:5320] = 0.1

    prepared = processor.prepare_feature_audio(waveform, 4000, 0.015)

    window_samples = round(BEATS_FEATURE_WINDOW_SECONDS * processor.sample_rate)
    pre_onset_samples = round(BEATS_PRE_ONSET_SECONDS * processor.sample_rate)
    detected_onset = 4800
    expected_start = detected_onset - pre_onset_samples
    assert processor.feature_window_bounds(waveform, 4000, 0.015) == (
        expected_start,
        detected_onset,
        expected_start + window_samples,
    )
    assert prepared.shape == (window_samples,)
    np.testing.assert_array_equal(
        prepared,
        waveform[expected_start : expected_start + window_samples],
    )


def test_beats_onset_window_zero_pads_recording_boundaries() -> None:
    processor = SignalProcessor(model="beats")
    window_samples = round(BEATS_FEATURE_WINDOW_SECONDS * processor.sample_rate)
    pre_onset_samples = round(BEATS_PRE_ONSET_SECONDS * processor.sample_rate)

    left_waveform = np.zeros(12000, dtype=np.float32)
    left_waveform[:320] = 0.1
    left = processor.prepare_feature_audio(left_waveform, 0, 0.015)
    np.testing.assert_array_equal(left[:pre_onset_samples], 0.0)
    np.testing.assert_array_equal(
        left[pre_onset_samples:], left_waveform[: window_samples - pre_onset_samples]
    )

    right_waveform = np.zeros(6000, dtype=np.float32)
    right_waveform[5000:5320] = 0.1
    right = processor.prepare_feature_audio(right_waveform, 4000, 0.015)
    source_start = 4800 - pre_onset_samples
    copied = right_waveform.size - source_start
    np.testing.assert_array_equal(right[:copied], right_waveform[source_start:])
    np.testing.assert_array_equal(right[copied:], 0.0)


def test_beats_onset_window_requires_threshold_crossing() -> None:
    processor = SignalProcessor(model="beats")

    with pytest.raises(ValueError, match="no threshold-qualified onset"):
        processor.prepare_feature_audio(np.zeros(8000, dtype=np.float32), 0, 0.015)


@pytest.mark.parametrize("model", ["legacy", "pann"])
def test_non_beats_feature_audio_remains_full_length(model: str) -> None:
    processor = SignalProcessor(model=model)
    waveform = np.linspace(-0.1, 0.1, 1000, dtype=np.float32)

    prepared = processor.prepare_feature_audio(waveform, 500, 0.015)

    np.testing.assert_array_equal(prepared, waveform)


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
    assert processor.checkpoint_path is None


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


def test_beats_is_default_and_uses_16khz() -> None:
    processor = SignalProcessor()

    assert processor.model == "beats"
    assert processor.sample_rate == 16000
    assert SignalProcessor(model="beats", sample_rate=8000).sample_rate == 16000
    assert processor.checkpoint_path == BEATS_CHECKPOINT_PATH
    assert SignalProcessor(model="default").model == "beats"


def test_prepare_eagerly_loads_and_caches_beats(monkeypatch) -> None:
    processor = SignalProcessor(model="beats")
    load_calls: list[str] = []

    def load_beats(checkpoint_path: str) -> None:
        load_calls.append(checkpoint_path)
        processor._beats_model = object()
        processor.checkpoint_sha256 = "test-sha256"

    monkeypatch.setattr(processor, "_load_beats", load_beats)

    assert processor.prepare() == "test-sha256"
    assert processor.prepare() == "test-sha256"
    assert load_calls == [str(processor.checkpoint_path)]


def test_beats_checkpoint_sidecar_is_enforced(tmp_path) -> None:
    checkpoint = tmp_path / "beats.pt"
    checkpoint.write_bytes(b"trusted-beats")
    processor = SignalProcessor(model="beats", checkpoint_path=checkpoint)

    with pytest.raises(FileNotFoundError, match="sidecar"):
        processor.validate_beats_checkpoint()

    checkpoint.with_suffix(".sha256").write_text("")
    with pytest.raises(ValueError, match="invalid"):
        processor.validate_beats_checkpoint()

    digest = processor._sha256_file(checkpoint)
    checkpoint.with_suffix(".sha256").write_text(f"{digest}  beats.pt\n")
    assert processor.validate_beats_checkpoint() == digest


def test_beats_rejects_malformed_checkpoint(tmp_path) -> None:
    import torch

    checkpoint = tmp_path / "beats.pt"
    torch.save({"unexpected": {}}, checkpoint)
    processor = SignalProcessor(
        model="beats",
        checkpoint_path=checkpoint,
        verify_checkpoint=False,
    )

    with pytest.raises(ValueError, match="must contain 'cfg' and 'model'"):
        processor.prepare()


def test_beats_extracts_deterministic_mean_pooled_embedding(monkeypatch) -> None:
    import torch

    class FakeBEATs:
        def __init__(self) -> None:
            self.patch_embedding = torch.nn.Conv2d(1, BEATS_EMBEDDING_DIM, 1, bias=False)
            torch.nn.init.constant_(self.patch_embedding.weight, 0.5)
            self.layer_norm = torch.nn.Identity()
            self.post_extract_proj = None
            self.dropout_input = torch.nn.Identity()

        @staticmethod
        def preprocess(waveform):
            return waveform[:, :16].reshape(1, 4, 4)

        @staticmethod
        def encoder(features):
            return features, []

    processor = SignalProcessor(model="beats")

    def load_beats(_checkpoint_path: str) -> None:
        processor._beats_model = FakeBEATs()
        processor._beats_device = torch.device("cpu")

    monkeypatch.setattr(processor, "_load_beats", load_beats)
    waveform = np.linspace(-1.0, 1.0, 16000, dtype=np.float32)

    first = processor.extract_features(waveform)
    second = processor.extract_features(waveform)

    assert first.shape == (BEATS_EMBEDDING_DIM,)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
