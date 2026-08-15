"""Canonical signal processor import.

The implementation remains in ``new_processor`` for import compatibility with
older callers. New code should import :class:`SignalProcessor` from this module.
"""

from src.audio.new_processor import (
    PANN_CHECKPOINT_PATH,
    PANN_CHECKPOINT_SHA256,
    SignalProcessor,
)

__all__ = ["PANN_CHECKPOINT_PATH", "PANN_CHECKPOINT_SHA256", "SignalProcessor"]
