import csv

import numpy as np
import pytest

from src.ml.dataset import REQUIRED_COLUMNS, load_manifest
from src.ml.evaluation import (
    calibrate_classifier,
    calibrate_grouped_classifier,
    evaluate_classifier,
)


def write_manifest(path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def row(path, label, subject, split) -> dict[str, str]:
    return {
        "path": str(path),
        "label": label,
        "subject": subject,
        "session": "session-1",
        "device": "mic-1",
        "sample_rate": "16000",
        "split": split,
    }


def test_manifest_accepts_disjoint_groups(tmp_path) -> None:
    files = [tmp_path / f"{index}.wav" for index in range(3)]
    for audio_file in files:
        audio_file.touch()
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            row(files[0], "desk", "subject-1", "train"),
            row(files[1], "desk", "subject-2", "validation"),
            row(files[2], "desk", "subject-3", "test"),
        ],
    )
    assert len(load_manifest(manifest)) == 3


def test_manifest_rejects_group_leakage(tmp_path) -> None:
    files = [tmp_path / f"{index}.wav" for index in range(3)]
    for audio_file in files:
        audio_file.touch()
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            row(files[0], "desk", "subject-1", "train"),
            row(files[1], "desk", "subject-1", "validation"),
            row(files[2], "desk", "subject-2", "test"),
        ],
    )
    with pytest.raises(ValueError, match="leak"):
        load_manifest(manifest)


def test_calibration_and_evaluation_separate_unknown_rejection() -> None:
    train_x = np.array(
        [[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0], [10.0, 10.0], [10.1, 10.0], [9.9, 10.0]]
    )
    train_y = np.array(["desk", "desk", "desk", "wall", "wall", "wall"])
    validation_x = np.array([[0.05, 0.0], [10.05, 10.0], [50.0, 50.0]])
    validation_y = np.array(["desk", "wall", "unknown-surface"])
    classifier, calibration = calibrate_classifier(
        train_x,
        train_y,
        validation_x,
        validation_y,
        neighbor_candidates=(1, 3),
    )

    report = evaluate_classifier(classifier, validation_x, validation_y)

    assert calibration.known_end_to_end_accuracy == 1.0
    assert calibration.unknown_rejection_rate == 1.0
    assert report.known_end_to_end_accuracy == 1.0
    assert report.unknown_rejection_rate == 1.0


def test_cosine_calibration_matches_runtime_distance_mode() -> None:
    train_x = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.01],
            [1.0, -0.01],
            [0.0, 1.0],
            [0.01, 1.0],
            [-0.01, 1.0],
        ]
    )
    train_y = np.array(["desk", "desk", "desk", "wall", "wall", "wall"])
    validation_x = np.array([[10.0, 0.0], [0.0, 10.0], [-1.0, 0.0]])
    validation_y = np.array(["desk", "wall", "unknown-surface"])

    classifier, calibration = calibrate_classifier(
        train_x,
        train_y,
        validation_x,
        validation_y,
        neighbor_candidates=(1,),
        distance_mode="cosine",
    )
    report = evaluate_classifier(classifier, validation_x, validation_y)

    assert classifier.distance_mode == "cosine"
    assert calibration.distance_thresholds == classifier.distance_thresholds
    assert calibration.known_end_to_end_accuracy == 1.0
    assert calibration.unknown_rejection_rate == 1.0
    assert report.known_end_to_end_accuracy == 1.0
    assert report.unknown_rejection_rate == 1.0


def test_calibration_requires_three_examples_per_class() -> None:
    with pytest.raises(ValueError, match="at least three"):
        calibrate_classifier(
            np.array([[0.0, 0.0], [0.1, 0.0]]),
            np.array(["desk", "desk"]),
            np.array([[0.05, 0.0]]),
            np.array(["desk"]),
        )


def test_grouped_calibration_uses_physical_captures_and_unknowns() -> None:
    features = []
    labels = []
    capture_ids = []
    augmented = []
    for label, center, first_id in (("desk", 0.0, 0), ("wall", 1.0, 4)):
        for offset in range(4):
            capture_id = first_id + offset
            vector = np.array([1.0 - center, center + offset * 0.002])
            for scale, is_augmented in ((1.0, False), (0.8, True)):
                features.append(vector * scale)
                labels.append(label)
                capture_ids.append(capture_id)
                augmented.append(is_augmented)
    negatives = np.tile(np.array([[-1.0, 0.0]]), (10, 1))

    classifier, result = calibrate_grouped_classifier(
        np.asarray(features),
        np.asarray(labels),
        np.asarray(capture_ids),
        np.asarray(augmented),
        negatives,
        distance_mode="cosine",
        neighbor_candidates=(1,),
        prototype_weight_candidates=(0.5,),
        spread_multiplier_candidates=(2.0,),
        margin_ratio_candidates=(0.1,),
        temperature_ratio_candidates=(0.5,),
    )

    assert classifier.class_capture_counts() == {"desk": 4, "wall": 4}
    assert result.fold_count == 4
    assert result.known_end_to_end_accuracy >= 0.75
    assert result.unknown_rejection_rate == 1.0
