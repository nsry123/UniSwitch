"""Evaluate knock classification from a leakage-checked dataset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np

from src.audio.augmentation import (
    DEFAULT_GAIN_DB,
    apply_gain_db,
    validate_gain_db,
    validate_gain_values,
)
from src.audio.processor import (
    BEATS_CHECKPOINT_PATH,
    BEATS_EMBEDDING_DIM,
    BEATS_FEATURE_ALIGNMENT,
    BEATS_FEATURE_WINDOW_SECONDS,
    BEATS_POOLING,
    BEATS_PRE_ONSET_SECONDS,
    BEATS_SOURCE_REVISION,
    DEFAULT_ENERGY_THRESHOLD,
    PANN_CHECKPOINT_PATH,
    SignalProcessor,
    beats_input_normalization_metadata,
)
from src.ml.dataset import AudioRecord, load_manifest
from src.ml.evaluation import calibrate_classifier, evaluate_classifier


def extract_split(
    records: list[AudioRecord],
    split: str,
    processor: SignalProcessor,
    gain_db_values: tuple[float, ...] = (),
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract deterministic features for one manifest split."""
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    for record in records:
        if record.split != split:
            continue
        waveform, native_sample_rate = librosa.load(record.path, sr=None, mono=True)
        if native_sample_rate != record.sample_rate:
            raise ValueError(
                f"manifest sample rate for {record.path} is {record.sample_rate}, "
                f"but the file reports {native_sample_rate}"
            )
        if native_sample_rate != processor.sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=native_sample_rate,
                target_sr=processor.sample_rate,
            )
        trimmed, _ = librosa.effects.trim(waveform, top_db=20)
        if trimmed.size == 0:
            raise ValueError(f"audio is empty after trimming: {record.path}")
        feature_audio = processor.prepare_feature_audio(trimmed, 0, energy_threshold)
        vectors.append(processor.extract_features(feature_audio))
        labels.append(record.label)
        for gain_db in gain_db_values:
            vectors.append(processor.extract_features(apply_gain_db(feature_audio, gain_db)))
            labels.append(record.label)
    return np.vstack(vectors), np.asarray(labels, dtype=np.str_)


def gain_db_argument(value: str) -> float:
    """Parse one CLI gain value with an argparse-compatible error."""
    try:
        return validate_gain_db(float(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def resolve_gain_db(arguments: argparse.Namespace) -> tuple[float, ...]:
    """Resolve default, custom, or disabled gain augmentation settings."""
    if arguments.model == "beats":
        if arguments.gain_db is not None:
            raise ValueError(
                "--gain-db is not supported for BEATs because input normalization "
                "already removes scalar gain"
            )
        return ()
    if arguments.no_gain_augmentation:
        return ()
    requested = arguments.gain_db
    return validate_gain_values(DEFAULT_GAIN_DB if requested is None else requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=("beats", "legacy", "pann"), default="beats")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--skip-checkpoint-verification", action="store_true")
    parser.add_argument("--neighbors", type=int, nargs="+", default=[1, 3, 5, 7])
    parser.add_argument("--energy-threshold", type=float, default=DEFAULT_ENERGY_THRESHOLD)
    parser.add_argument("--save-model", type=Path)
    gain_group = parser.add_mutually_exclusive_group()
    gain_group.add_argument(
        "--gain-db",
        action="append",
        type=gain_db_argument,
        help=(
            "gain in dB to add as a PANN/legacy training variant; repeat for "
            "multiple values (unsupported for BEATs)"
        ),
    )
    gain_group.add_argument("--no-gain-augmentation", action="store_true")
    return parser.parse_args()


def main(arguments: argparse.Namespace) -> int:
    records = load_manifest(arguments.manifest)
    gain_db_values = resolve_gain_db(arguments)
    checkpoint_path = arguments.checkpoint
    if checkpoint_path is None:
        checkpoint_path = {
            "beats": BEATS_CHECKPOINT_PATH,
            "pann": PANN_CHECKPOINT_PATH,
        }.get(arguments.model)
    processor = SignalProcessor(
        model=arguments.model,
        checkpoint_path=checkpoint_path,
        verify_checkpoint=not arguments.skip_checkpoint_verification,
    )
    checkpoint_sha256 = None
    if arguments.model in {"beats", "pann"}:
        checkpoint_sha256 = processor.prepare()

    train_features, train_labels = extract_split(
        records,
        "train",
        processor,
        gain_db_values,
        arguments.energy_threshold,
    )
    validation_features, validation_labels = extract_split(
        records,
        "validation",
        processor,
        energy_threshold=arguments.energy_threshold,
    )
    test_features, test_labels = extract_split(
        records,
        "test",
        processor,
        energy_threshold=arguments.energy_threshold,
    )
    classifier, calibration = calibrate_classifier(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        arguments.neighbors,
        distance_mode=(
            "cosine" if arguments.model in {"beats", "pann"} else "standardized_euclidean"
        ),
    )
    report = evaluate_classifier(classifier, test_features, test_labels)
    result = {"calibration": calibration.__dict__, "test": report.to_dict()}
    print(json.dumps(result, indent=2, sort_keys=True))

    if arguments.save_model is not None:
        manifest_digest = hashlib.sha256(arguments.manifest.read_bytes()).hexdigest()
        metadata = {
            "feature_extractor": arguments.model,
            "checkpoint_sha256": checkpoint_sha256,
            "sample_rate": processor.sample_rate,
            "dataset_manifest_sha256": manifest_digest,
            "calibration": calibration.__dict__,
            "gain_augmentation": {
                "gain_db": list(gain_db_values),
                "peak_safe": True,
            },
        }
        if arguments.model == "beats":
            metadata.update(
                {
                    "embedding_dim": BEATS_EMBEDDING_DIM,
                    "pooling": BEATS_POOLING,
                    "source_revision": BEATS_SOURCE_REVISION,
                    "feature_window": {
                        "alignment": BEATS_FEATURE_ALIGNMENT,
                        "duration_seconds": BEATS_FEATURE_WINDOW_SECONDS,
                        "pre_onset_seconds": BEATS_PRE_ONSET_SECONDS,
                    },
                    "energy_threshold": arguments.energy_threshold,
                    "input_normalization": beats_input_normalization_metadata(),
                }
            )
        classifier.save(
            arguments.save_model,
            metadata=metadata,
        )
        print(f"Saved calibrated classifier to {arguments.save_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
