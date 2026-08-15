from pathlib import Path

import numpy as np
import pytest

from src.audio.processor import PANN_CHECKPOINT_PATH, PANN_CHECKPOINT_SHA256, SignalProcessor


@pytest.mark.integration
def test_pann_checkpoint_is_compatible_and_deterministic(monkeypatch) -> None:
    checkpoint = Path(PANN_CHECKPOINT_PATH)
    if not checkpoint.is_file():
        pytest.skip("optional PANN checkpoint is not present")

    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    processor = SignalProcessor(model="pann", checkpoint_path=checkpoint)
    waveform = np.zeros(int(processor.sample_rate * 1.7), dtype=np.float32)
    waveform[5000:5050] = np.hanning(50).astype(np.float32)

    first = processor.extract_features(waveform)
    second = processor.extract_features(waveform)

    assert processor.checkpoint_sha256 == PANN_CHECKPOINT_SHA256
    assert first.shape == (2048,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
