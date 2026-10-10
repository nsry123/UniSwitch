import numpy as np
import pytest

from src.audio.augmentation import apply_gain_db, validate_gain_db


def test_negative_gain_scales_without_mutating_input() -> None:
    waveform = np.array([-0.5, 0.25], dtype=np.float32)
    original = waveform.copy()

    augmented = apply_gain_db(waveform, -6.0)

    np.testing.assert_allclose(augmented, waveform * (10.0 ** (-6.0 / 20.0)))
    np.testing.assert_array_equal(waveform, original)
    assert augmented.dtype == waveform.dtype
    assert augmented.shape == waveform.shape


def test_positive_gain_is_limited_to_prevent_clipping() -> None:
    waveform = np.array([-0.75, 0.5], dtype=np.float64)

    augmented = apply_gain_db(waveform, 6.0)

    assert np.max(np.abs(augmented)) == pytest.approx(1.0)
    np.testing.assert_allclose(augmented, waveform / 0.75)


def test_silence_remains_silent() -> None:
    waveform = np.zeros(8, dtype=np.float32)

    augmented = apply_gain_db(waveform, 6.0)

    np.testing.assert_array_equal(augmented, waveform)
    assert augmented is not waveform


@pytest.mark.parametrize(
    ("waveform", "gain_db", "error_type"),
    [
        (np.array([], dtype=np.float32), -9.0, ValueError),
        (np.zeros((2, 2), dtype=np.float32), -9.0, ValueError),
        (np.array([0, 1], dtype=np.int16), -9.0, TypeError),
        (np.array([0.0, np.nan], dtype=np.float32), -9.0, ValueError),
        (np.ones(2, dtype=np.float32), 0.0, ValueError),
        (np.ones(2, dtype=np.float32), np.inf, ValueError),
    ],
)
def test_gain_rejects_invalid_inputs(waveform, gain_db, error_type) -> None:
    with pytest.raises(error_type):
        apply_gain_db(waveform, gain_db)


def test_gain_validator_rejects_duplicate_original() -> None:
    with pytest.raises(ValueError, match="original"):
        validate_gain_db(0.0)
