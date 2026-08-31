from pathlib import Path

import numpy as np

import scripts.evaluate as evaluate
from src.ml.dataset import AudioRecord


class RecordingProcessor:
    sample_rate = 16000

    def __init__(self) -> None:
        self.waveforms: list[np.ndarray] = []

    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        self.waveforms.append(waveform.copy())
        return np.array([waveform.mean(), waveform.max()])

    def prepare_feature_audio(
        self,
        waveform: np.ndarray,
        trigger_sample: int,
        energy_threshold: float = 0.015,
    ) -> np.ndarray:
        return waveform


def record(split: str) -> AudioRecord:
    return AudioRecord(
        path=Path(f"{split}.wav"),
        label="desk",
        subject=f"subject-{split}",
        session="session-1",
        device="mic-1",
        sample_rate=16000,
        split=split,
    )


def test_train_split_adds_ordered_gain_variants(monkeypatch) -> None:
    waveform = np.array([0.25, -0.5, 0.0], dtype=np.float32)
    monkeypatch.setattr(evaluate.librosa, "load", lambda *_args, **_kwargs: (waveform, 16000))
    monkeypatch.setattr(evaluate.librosa.effects, "trim", lambda audio, **_kwargs: (audio, None))
    processor = RecordingProcessor()

    features, labels = evaluate.extract_split(
        [record("train")],
        "train",
        processor,
        (-6.0, 6.0),
    )

    assert features.shape == (3, 2)
    assert labels.tolist() == ["desk", "desk", "desk"]
    np.testing.assert_array_equal(processor.waveforms[0], waveform)
    np.testing.assert_allclose(processor.waveforms[1], waveform * (10.0 ** (-6.0 / 20.0)))
    np.testing.assert_allclose(processor.waveforms[2], waveform * (10.0 ** (6.0 / 20.0)))


def test_validation_and_test_are_unaugmented_when_no_recipe_is_passed(monkeypatch) -> None:
    waveform = np.array([0.25, -0.5], dtype=np.float32)
    monkeypatch.setattr(evaluate.librosa, "load", lambda *_args, **_kwargs: (waveform, 16000))
    monkeypatch.setattr(evaluate.librosa.effects, "trim", lambda audio, **_kwargs: (audio, None))

    for split in ("validation", "test"):
        processor = RecordingProcessor()
        features, labels = evaluate.extract_split([record(split)], split, processor)
        assert features.shape == (1, 2)
        assert labels.tolist() == ["desk"]
        assert len(processor.waveforms) == 1


def test_split_prepares_once_before_applying_gain(monkeypatch) -> None:
    waveform = np.array([0.0, 0.5, 0.25, 0.0], dtype=np.float32)
    monkeypatch.setattr(evaluate.librosa, "load", lambda *_args, **_kwargs: (waveform, 16000))
    monkeypatch.setattr(evaluate.librosa.effects, "trim", lambda audio, **_kwargs: (audio, None))

    class CroppingProcessor(RecordingProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.prepare_calls: list[tuple[int, float]] = []

        def prepare_feature_audio(
            self,
            audio: np.ndarray,
            trigger_sample: int,
            energy_threshold: float = 0.015,
        ) -> np.ndarray:
            self.prepare_calls.append((trigger_sample, energy_threshold))
            return audio[1:3]

    processor = CroppingProcessor()
    evaluate.extract_split(
        [record("train")],
        "train",
        processor,
        (-6.0,),
        energy_threshold=0.02,
    )

    assert processor.prepare_calls == [(0, 0.02)]
    np.testing.assert_array_equal(processor.waveforms[0], waveform[1:3])
    np.testing.assert_allclose(
        processor.waveforms[1], waveform[1:3] * (10.0 ** (-6.0 / 20.0))
    )
