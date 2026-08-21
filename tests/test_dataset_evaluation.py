import csv

import numpy as np
import pytest

from src.ml.classifier import TapClassifier
from src.ml.dataset import REQUIRED_COLUMNS, load_manifest
from src.ml.evaluation import calibrate_classifier, evaluate_classifier


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
        [[1.0, 0.0], [1.0, 0.01], [0.0, 1.0], [0.01, 1.0]]
    )
    train_y = np.array(["desk", "desk", "wall", "wall"])
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


def test_auto_calibration_uses_validation_impostors_per_label() -> None:
    train_x = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5)
    train_y = np.array(["desk"] * 5 + ["wall"] * 5)
    validation_x = np.array(
        [
            [1.0, 0.01],
            [0.01, 1.0],
            [0.8, 0.6],
            [0.6, 0.8],
        ]
    )
    validation_y = np.array(["desk", "wall", "unknown-a", "unknown-b"])

    classifier, calibration = calibrate_classifier(
        train_x,
        train_y,
        validation_x,
        validation_y,
        neighbor_candidates=(1,),
        distance_mode="cosine",
        threshold_mode="auto",
        distance_threshold_cap=0.2,
    )
    report = evaluate_classifier(classifier, validation_x, validation_y)

    assert classifier.threshold_calibrations["desk"].source == "validation"
    assert classifier.threshold_calibrations["wall"].source == "validation"
    assert classifier.separation_calibrations["desk"].source == "validation"
    assert classifier.separation_calibrations["wall"].source == "validation"
    assert calibration.distance_thresholds == classifier.distance_thresholds
    assert calibration.separation_thresholds == classifier.separation_thresholds
    assert report.known_end_to_end_accuracy == 1.0
    assert report.unknown_rejection_rate == 1.0
    assert report.known_confusion_matrix["desk"]["desk"] == 1
    assert report.per_label_recall == {"desk": 1.0, "wall": 1.0}
    assert report.separation_summary["desk"]["count"] > 0


def test_evaluation_reports_ambiguity_rejections() -> None:
    classifier = TapClassifier(
        n_neighbors=1,
        distance_threshold=1.0,
        distance_mode="cosine",
        separation_threshold=0.15,
    )
    classifier.register_tap(np.array([1.0, 0.0]), "left")
    classifier.register_tap(np.array([np.cos(0.1), np.sin(0.1)]), "right")
    features = np.array([[np.cos(0.05), np.sin(0.05)], [1.0, 0.0]])
    labels = np.array(["left", "left"])

    report = evaluate_classifier(classifier, features, labels)

    assert report.ambiguity_rejection_rate == 0.5
    assert report.known_confusion_matrix["left"]["Unknown"] == 1
    assert report.known_accepted_accuracy == 1.0
