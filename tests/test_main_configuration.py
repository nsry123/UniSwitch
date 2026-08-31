import sys

import numpy as np
import pytest

from main import (
    DEFAULT_DISTANCE_THRESHOLDS,
    load_or_create_classifier,
    parse_args,
    resolve_checkpoint_path,
    resolve_gain_db,
    resolve_state_path,
)
from scripts.evaluate import parse_args as parse_evaluate_args
from src.audio.processor import BEATS_CHECKPOINT_PATH, PANN_CHECKPOINT_PATH
from src.ml.classifier import TapClassifier


def test_beats_is_cli_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py"])

    arguments = parse_args()

    assert arguments.model == "beats"
    assert arguments.checkpoint is None
    assert arguments.state_path is None
    assert DEFAULT_DISTANCE_THRESHOLDS["beats"] == 0.15
    assert resolve_gain_db(arguments) == ()


def test_gain_cli_accepts_custom_values_and_disable(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--model", "pann", "--gain-db", "-3", "--gain-db", "4"],
    )
    custom = parse_args()
    assert resolve_gain_db(custom) == (-3.0, 4.0)

    monkeypatch.setattr(
        sys, "argv", ["main.py", "--model", "legacy", "--no-gain-augmentation"]
    )
    disabled = parse_args()
    assert resolve_gain_db(disabled) == ()


def test_beats_rejects_explicit_gain_augmentation(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py", "--gain-db", "-6"])

    with pytest.raises(ValueError, match="not supported for BEATs"):
        resolve_gain_db(parse_args())


def test_model_specific_checkpoint_and_state_paths() -> None:
    assert resolve_checkpoint_path("beats", None) == BEATS_CHECKPOINT_PATH
    assert resolve_checkpoint_path("pann", None) == PANN_CHECKPOINT_PATH
    assert resolve_checkpoint_path("legacy", None) is None
    assert (
        resolve_state_path("beats", None).name
        == "beats-onset200-normalized-registrations.npz"
    )
    assert resolve_state_path("pann", None).name == "registrations.npz"
    assert resolve_state_path("legacy", None).name == "legacy-registrations.npz"


def test_fresh_beats_classifier_uses_provisional_defaults() -> None:
    classifier = load_or_create_classifier(
        None,
        "beats",
        {
            "feature_extractor": "beats",
            "checkpoint_sha256": "beats-sha",
            "sample_rate": 16000,
            "embedding_dim": 768,
            "pooling": "mean",
            "source_revision": "revision",
        },
        3,
        DEFAULT_DISTANCE_THRESHOLDS["beats"],
        "cosine",
        0.5,
        2.0,
        None,
        None,
    )

    assert classifier.distance_threshold == 0.15
    assert classifier.margin_threshold == pytest.approx(0.015)
    assert classifier.probability_temperature == pytest.approx(0.075)


def test_beats_is_evaluator_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--manifest", "dataset.csv"])

    arguments = parse_evaluate_args()

    assert arguments.model == "beats"
    assert arguments.checkpoint is None
    assert arguments.energy_threshold == 0.015


def test_pann_state_remains_available_only_for_pann(tmp_path) -> None:
    state_path = tmp_path / "registrations.npz"
    classifier = TapClassifier(n_neighbors=1, distance_threshold=0.2, distance_mode="cosine")
    for offset in (0.0, 0.01, -0.01):
        classifier.register_tap(np.array([1.0, offset]), "desk")
    pann_metadata = {
        "feature_extractor": "pann",
        "checkpoint_sha256": "pann-sha",
        "sample_rate": 32000,
    }
    classifier.save(state_path, pann_metadata)

    current_metadata = {
        **pann_metadata,
        "gain_augmentation": {"gain_db": [-9.0, 6.0], "peak_safe": True},
    }
    restored = load_or_create_classifier(
        state_path,
        "pann",
        current_metadata,
        1,
        0.2,
        "cosine",
        0.5,
        2.0,
        None,
        None,
    )
    assert restored.metadata["feature_extractor"] == "pann"

    with pytest.raises(ValueError, match="feature extractor"):
        load_or_create_classifier(
            state_path,
            "beats",
            {
                "feature_extractor": "beats",
                "checkpoint_sha256": "beats-sha",
                "sample_rate": 16000,
                "embedding_dim": 768,
                "pooling": "mean",
                "source_revision": "revision",
            },
            1,
            0.15,
            "cosine",
            0.5,
            2.0,
            None,
            None,
        )


def test_beats_state_rejects_pooling_contract_change(tmp_path) -> None:
    state_path = tmp_path / "beats.npz"
    classifier = TapClassifier(n_neighbors=1, distance_threshold=0.15, distance_mode="cosine")
    for offset in (0.0, 0.01, -0.01):
        classifier.register_tap(np.array([1.0, offset]), "desk")
    metadata = {
        "feature_extractor": "beats",
        "checkpoint_sha256": "beats-sha",
        "sample_rate": 16000,
        "embedding_dim": 768,
        "pooling": "mean",
        "source_revision": "revision",
    }
    classifier.save(state_path, metadata)

    changed = {**metadata, "pooling": "max"}
    with pytest.raises(ValueError, match="pooling"):
        load_or_create_classifier(
            state_path,
            "beats",
            changed,
            1,
            0.15,
            "cosine",
            0.5,
            2.0,
            None,
            None,
        )


def test_beats_state_rejects_missing_onset_window_contract(tmp_path) -> None:
    state_path = tmp_path / "old-beats.npz"
    classifier = TapClassifier(n_neighbors=1, distance_threshold=0.15, distance_mode="cosine")
    for offset in (0.0, 0.01, -0.01):
        classifier.register_tap(np.array([1.0, offset]), "desk")
    old_metadata = {
        "feature_extractor": "beats",
        "checkpoint_sha256": "beats-sha",
        "sample_rate": 16000,
        "embedding_dim": 768,
        "pooling": "mean",
        "source_revision": "revision",
    }
    classifier.save(state_path, old_metadata)
    current_metadata = {
        **old_metadata,
        "feature_window": {
            "alignment": "first_threshold_onset",
            "duration_seconds": 0.58,
            "pre_onset_seconds": 0.2,
        },
        "energy_threshold": 0.015,
    }

    with pytest.raises(ValueError, match="re-register using the onset-aligned"):
        load_or_create_classifier(
            state_path,
            "beats",
            current_metadata,
            1,
            0.15,
            "cosine",
            0.5,
            2.0,
            None,
            None,
        )

    compatible_path = tmp_path / "onset-beats.npz"
    classifier.save(compatible_path, current_metadata)
    restored = load_or_create_classifier(
        compatible_path,
        "beats",
        current_metadata,
        1,
        0.15,
        "cosine",
        0.5,
        2.0,
        None,
        None,
    )
    assert restored.metadata["feature_window"] == current_metadata["feature_window"]
    assert restored.metadata["energy_threshold"] == 0.015


def test_beats_state_rejects_missing_normalization_contract(tmp_path) -> None:
    state_path = tmp_path / "unnormalized-beats.npz"
    classifier = TapClassifier(n_neighbors=1, distance_threshold=0.15, distance_mode="cosine")
    classifier.register_tap(np.array([1.0, 0.0]), "desk")
    metadata = {
        "feature_extractor": "beats",
        "checkpoint_sha256": "beats-sha",
        "sample_rate": 16000,
        "embedding_dim": 768,
        "pooling": "mean",
        "source_revision": "revision",
        "feature_window": {
            "alignment": "first_threshold_onset",
            "duration_seconds": 0.58,
            "pre_onset_seconds": 0.2,
        },
        "energy_threshold": 0.015,
    }
    classifier.save(state_path, metadata)

    with pytest.raises(ValueError, match="re-register with normalized BEATs"):
        load_or_create_classifier(
            state_path,
            "beats",
            {**metadata, "input_normalization": {"version": "test-v1"}},
            1,
            0.15,
            "cosine",
            0.5,
            2.0,
            None,
            None,
        )
