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


@pytest.mark.integration
def test_beats_embedding_is_invariant_to_scalar_gain() -> None:
    checkpoint = Path(BEATS_CHECKPOINT_PATH)
    sidecar = checkpoint.with_suffix(".sha256")
    if not checkpoint.is_file() or not sidecar.is_file():
        pytest.skip("optional verified BEATs checkpoint is not present")

    processor = SignalProcessor(model="beats", checkpoint_path=checkpoint)
    waveform = np.zeros(round(0.58 * processor.sample_rate), dtype=np.float32)
    impact = np.hanning(480).astype(np.float32)
    waveform[3200:3680] = 0.15 * impact * np.sin(
        2 * np.pi * 900 * np.arange(480) / processor.sample_rate
    )
    embeddings = []
    for gain_db in (-12.0, -6.0, 0.0, 6.0):
        scaled = waveform * np.float32(10.0 ** (gain_db / 20.0))
        embedding = processor.extract_features(scaled)
        embeddings.append(embedding / np.linalg.norm(embedding))

    reference = embeddings[2]
    for embedding in embeddings:
        assert 1.0 - float(reference @ embedding) < 1e-6


@pytest.mark.integration
def test_beats_normalization_preserves_spectral_distinctions() -> None:
    checkpoint = Path(BEATS_CHECKPOINT_PATH)
    sidecar = checkpoint.with_suffix(".sha256")
    if not checkpoint.is_file() or not sidecar.is_file():
        pytest.skip("optional verified BEATs checkpoint is not present")

    processor = SignalProcessor(model="beats", checkpoint_path=checkpoint)
    sample_count = round(0.58 * processor.sample_rate)
    time = np.arange(sample_count, dtype=np.float32) / processor.sample_rate
    envelope = np.exp(-20 * np.maximum(time - 0.2, 0.0)).astype(np.float32)
    envelope[time < 0.2] = 0.0
    low = envelope * np.sin(2 * np.pi * 250 * time)
    high = envelope * np.sin(2 * np.pi * 2500 * time)
    low_embedding = processor.extract_features(low)
    high_embedding = processor.extract_features(high)
    low_embedding /= np.linalg.norm(low_embedding)
    high_embedding /= np.linalg.norm(high_embedding)

    assert 1.0 - float(low_embedding @ high_embedding) > 1e-3
