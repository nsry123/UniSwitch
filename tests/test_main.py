import numpy as np

from main import classifier_status_lines
from src.ml.classifier import TapClassifier


def test_classifier_status_reports_per_label_calibration() -> None:
    classifier = TapClassifier(
        n_neighbors=1,
        distance_threshold=0.2,
        distance_mode="cosine",
        threshold_mode="auto",
    )
    for angle in (0.0, 0.1, 0.2, 0.3, 0.4):
        classifier.register_tap(np.array([np.cos(angle), np.sin(angle)]), "desk")

    status = "\n".join(classifier_status_lines(classifier))

    assert "desk: distance=" in status
    assert "separation=" in status
    assert "samples=5" in status
    assert "(registration)" in status
