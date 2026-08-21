"""Canonical signal processor import.

The implementation remains in ``new_processor`` for import compatibility with
older callers. New code should import :class:`SignalProcessor` from this module.
"""

from src.audio.new_processor import (
    BEATS_CHECKPOINT_PATH,
    BEATS_EMBEDDING_DIM,
    BEATS_POOLING,
    BEATS_SOURCE_REVISION,
    PANN_CHECKPOINT_PATH,
    PANN_CHECKPOINT_SHA256,
    SignalProcessor,
)

__all__ = [
    "BEATS_CHECKPOINT_PATH",
    "BEATS_EMBEDDING_DIM",
    "BEATS_POOLING",
    "BEATS_SOURCE_REVISION",
    "PANN_CHECKPOINT_PATH",
    "PANN_CHECKPOINT_SHA256",
    "SignalProcessor",
]
