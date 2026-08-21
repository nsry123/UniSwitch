import sys

import numpy as np
import pytest

from main import (
    DEFAULT_DISTANCE_THRESHOLDS,
    load_or_create_classifier,
    parse_args,
    resolve_checkpoint_path,
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


def test_model_specific_checkpoint_and_state_paths() -> None:
    assert resolve_checkpoint_path("beats", None) == BEATS_CHECKPOINT_PATH
    assert resolve_checkpoint_path("pann", None) == PANN_CHECKPOINT_PATH
    assert resolve_checkpoint_path("legacy", None) is None
    assert resolve_state_path("beats", None).name == "beats-registrations.npz"
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

    restored = load_or_create_classifier(
        state_path,
        "pann",
        pann_metadata,
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
