from pathlib import Path

import numpy as np
import pytest

from src.audio.processor import BEATS_CHECKPOINT_PATH, BEATS_EMBEDDING_DIM, SignalProcessor


@pytest.mark.integration
def test_beats_checkpoint_is_compatible_and_deterministic() -> None:
    checkpoint = Path(BEATS_CHECKPOINT_PATH)
    sidecar = checkpoint.with_suffix(".sha256")
    if not checkpoint.is_file() or not sidecar.is_file():
        pytest.skip("optional verified BEATs checkpoint is not present")

    processor = SignalProcessor(model="beats", checkpoint_path=checkpoint)
    waveform = np.zeros(int(processor.sample_rate * 1.7), dtype=np.float32)
    waveform[4000:4400] = 0.1

    first = processor.extract_features(waveform)
    second = processor.extract_features(waveform)

    assert first.shape == (BEATS_EMBEDDING_DIM,)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    assert processor.checkpoint_sha256 == sidecar.read_text().split()[0]
