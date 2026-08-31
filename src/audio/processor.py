"""Canonical signal processor import.

The implementation remains in ``new_processor`` for import compatibility with
older callers. New code should import :class:`SignalProcessor` from this module.
"""

from src.audio.new_processor import (
    BEATS_CHECKPOINT_PATH,
    BEATS_EMBEDDING_DIM,
    BEATS_FBANK_EPSILON,
    BEATS_FBANK_NORMALIZATION,
    BEATS_FBANK_STD_SCALE,
    BEATS_FEATURE_ALIGNMENT,
    BEATS_FEATURE_WINDOW_SECONDS,
    BEATS_NORMALIZATION_VERSION,
    BEATS_PEAK_CEILING,
    BEATS_POOLING,
    BEATS_PRE_ONSET_SECONDS,
    BEATS_SILENCE_EPSILON,
    BEATS_SOURCE_REVISION,
    BEATS_TARGET_RMS,
    DEFAULT_ENERGY_THRESHOLD,
    PANN_CHECKPOINT_PATH,
    PANN_CHECKPOINT_SHA256,
    SignalProcessor,
    beats_input_normalization_metadata,
)

__all__ = [
    "BEATS_CHECKPOINT_PATH",
    "BEATS_EMBEDDING_DIM",
    "BEATS_FEATURE_ALIGNMENT",
    "BEATS_FEATURE_WINDOW_SECONDS",
    "BEATS_FBANK_EPSILON",
    "BEATS_FBANK_NORMALIZATION",
    "BEATS_FBANK_STD_SCALE",
    "BEATS_NORMALIZATION_VERSION",
    "BEATS_PEAK_CEILING",
    "BEATS_POOLING",
    "BEATS_PRE_ONSET_SECONDS",
    "BEATS_SILENCE_EPSILON",
    "BEATS_SOURCE_REVISION",
    "BEATS_TARGET_RMS",
    "DEFAULT_ENERGY_THRESHOLD",
    "PANN_CHECKPOINT_PATH",
    "PANN_CHECKPOINT_SHA256",
    "SignalProcessor",
    "beats_input_normalization_metadata",
]
