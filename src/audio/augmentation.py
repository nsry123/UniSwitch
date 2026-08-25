"""Label-preserving waveform augmentation helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

DEFAULT_GAIN_DB: tuple[float, ...] = (-9.0, 6.0)


def validate_gain_db(gain_db: float) -> float:
    """Return a finite, nonzero gain value expressed in decibels."""
    value = float(gain_db)
    if not np.isfinite(value):
        raise ValueError("gain_db must be finite")
    if value == 0.0:
        raise ValueError("gain_db must be nonzero because the original is retained")
    return value


def validate_gain_values(gain_db_values: Iterable[float]) -> tuple[float, ...]:
    """Validate a gain recipe while preserving its deterministic order."""
    return tuple(validate_gain_db(value) for value in gain_db_values)


def apply_gain_db(waveform: np.ndarray, gain_db: float) -> np.ndarray:
    """Return a gain-adjusted mono waveform without clipping or input mutation."""
    audio = np.asarray(waveform)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"waveform must be a non-empty 1D array, got {audio.shape}")
    if not np.issubdtype(audio.dtype, np.floating):
        raise TypeError("waveform must use a floating-point dtype")
    if not np.isfinite(audio).all():
        raise ValueError("waveform contains NaN or infinite values")

    value = validate_gain_db(gain_db)
    scale = 10.0 ** (value / 20.0)
    peak = float(np.max(np.abs(audio)))
    if value > 0.0 and peak > 0.0:
        scale = min(scale, 1.0 / peak)

    return np.multiply(audio, scale, dtype=audio.dtype)
