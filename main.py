"""Command-line application for interactive knock registration and inference."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyaudio

from src.audio.augmentation import DEFAULT_GAIN_DB, validate_gain_db, validate_gain_values
from src.audio.listener import AudioListener
from src.audio.processor import (
    BEATS_CHECKPOINT_PATH,
    BEATS_EMBEDDING_DIM,
    BEATS_FEATURE_ALIGNMENT,
    BEATS_FEATURE_WINDOW_SECONDS,
    BEATS_POOLING,
    BEATS_PRE_ONSET_SECONDS,
    BEATS_SOURCE_REVISION,
    PANN_CHECKPOINT_PATH,
    SignalProcessor,
    beats_input_normalization_metadata,
)
from src.connectors import (
    DEFAULT_CONNECTOR_CONFIG_PATH,
    ActionEvent,
    ConnectorDispatcher,
    DeliveryResult,
    load_connectors,
)
from src.controller import EventDecision, EventStatus, SwitchController
from src.ml.classifier import DistanceMode, TapClassifier, ThresholdMode
from src.web.app import MicrophoneMonitor

DEFAULT_STATE_PATHS = {
    "beats": Path("state/beats-onset200-normalized-registrations.npz"),
    "pann": Path("state/registrations.npz"),
    "legacy": Path("state/legacy-registrations.npz"),
}
DEFAULT_CHECKPOINT_PATHS = {
    "beats": BEATS_CHECKPOINT_PATH,
    "pann": PANN_CHECKPOINT_PATH,
}
DEFAULT_DISTANCE_THRESHOLDS = {"beats": 0.15, "pann": 0.20, "legacy": 2.5}


def action_event_from_decision(decision: EventDecision) -> ActionEvent | None:
    """Convert only a confirmed controller event into a connector event."""
    if decision.status is not EventStatus.EVENT or decision.label is None:
        return None
    classification = decision.classification
    probabilities = () if classification is None else classification.probabilities.values()
    confidence = max(probabilities, default=None)
    return ActionEvent.create(decision.label, decision.tap_count, confidence)


def dispatch_decision(decision: EventDecision, dispatcher: ConnectorDispatcher) -> bool:
    """Submit an accepted event when at least one connector is configured."""
    if not dispatcher.connector_ids:
        return False
    event = action_event_from_decision(decision)
    return False if event is None else dispatcher.submit(event)


def report_connector_result(result: DeliveryResult) -> None:
    """Print an asynchronous delivery outcome without exposing credentials."""
    marker = "+" if result.success else "!"
    print(f"\n[{marker}] Connector {result.connector_id!r}: {result.message}")


def load_connector_dispatcher(arguments: argparse.Namespace) -> ConnectorDispatcher:
    """Load and health-check optional connectors without starting their workers."""
    requested_path = arguments.connector_config
    if arguments.no_connectors:
        connectors = []
    else:
        config_path = (
            DEFAULT_CONNECTOR_CONFIG_PATH if requested_path is None else requested_path
        )
        connectors = load_connectors(
            config_path,
            missing_ok=requested_path is None,
        )
    dispatcher = ConnectorDispatcher(connectors, result_callback=report_connector_result)
    if not connectors:
        print("[*] Smart-home connectors: disabled")
        return dispatcher
    health = dispatcher.check_health()
    print(f"[*] Smart-home connectors: {', '.join(dispatcher.connector_ids)}")
    for connector_id, result in zip(dispatcher.connector_ids, health, strict=True):
        marker = "+" if result.available else "!"
        print(f"[{marker}] Connector {connector_id!r}: {result.message}")
    return dispatcher


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


def resolve_state_path(model_name: str, requested: Path | None) -> Path:
    """Resolve an explicit or model-specific default registration path."""
    return requested if requested is not None else DEFAULT_STATE_PATHS[model_name]


def resolve_checkpoint_path(model_name: str, requested: Path | None) -> Path | None:
    """Resolve an explicit or model-specific neural checkpoint path."""
    if requested is not None:
        return requested
    return DEFAULT_CHECKPOINT_PATHS.get(model_name)


def list_input_devices() -> None:
    """Print available microphone devices and their native sample rates."""
    audio = pyaudio.PyAudio()
    try:
        print("--- Available Audio Input Devices ---\n")
        for index in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(index)
            if device_info.get("maxInputChannels", 0) > 0:
                print(f"Index [{index}] : {device_info.get('name')}")
                print(f"           - Max Channels: {device_info.get('maxInputChannels')}")
                print(
                    f"           - Default Rate: {int(device_info.get('defaultSampleRate'))} Hz\n"
                )
    finally:
        audio.terminate()


def load_or_create_classifier(
    state_path: Path | None,
    model_name: str,
    extractor_metadata: Mapping[str, Any],
    n_neighbors: int,
    distance_threshold: float,
    distance_mode: DistanceMode,
    prototype_weight: float,
    spread_multiplier: float,
    margin_threshold: float | None,
    probability_temperature: float | None,
    threshold_mode: ThresholdMode = "fixed",
) -> TapClassifier:
    """Load compatible registrations or return a new classifier."""
    if state_path is None or not state_path.exists():
        return TapClassifier(
            n_neighbors=n_neighbors,
            distance_threshold=distance_threshold,
            distance_mode=distance_mode,
            prototype_weight=prototype_weight,
            spread_multiplier=spread_multiplier,
            margin_threshold=margin_threshold,
            probability_temperature=probability_temperature,
            threshold_mode=threshold_mode,
        )

    classifier = TapClassifier.load(state_path)
    saved_model = classifier.metadata.get("feature_extractor")
    if saved_model != model_name:
        raise ValueError(
            f"state file uses feature extractor {saved_model!r}, not {model_name!r}; "
            "select the matching --model or a different --state-path"
        )
    if model_name in {"beats", "pann"} and classifier.metadata.get(
        "checkpoint_sha256"
    ) != extractor_metadata.get("checkpoint_sha256"):
        raise ValueError(
            f"state file checkpoint hash differs from the selected {model_name} checkpoint; "
            "use a compatible state file"
        )
    if model_name == "beats":
        for key in (
            "sample_rate",
            "embedding_dim",
            "pooling",
            "source_revision",
            "feature_window",
            "energy_threshold",
            "input_normalization",
        ):
            if classifier.metadata.get(key) != extractor_metadata.get(key):
                if key == "input_normalization":
                    raise ValueError(
                        "state file BEATs input_normalization differs from the selected "
                        "encoder; re-register with normalized BEATs features or select "
                        "a compatible state file"
                    )
                raise ValueError(
                    f"state file BEATs {key} differs from the selected encoder; "
                    "re-register using the onset-aligned default state path or "
                    "select a compatible state file"
                )
    if classifier.distance_mode != distance_mode:
        raise ValueError(
            f"state file uses distance mode {classifier.distance_mode!r}, "
            f"not {distance_mode!r}; use a compatible state file"
        )
    if classifier.threshold_mode != threshold_mode:
        raise ValueError(
            f"state file uses threshold mode {classifier.threshold_mode!r}, "
            f"not {threshold_mode!r}; use a compatible state file"
        )
    print(f"[*] Loaded {len(classifier.X_data)} registration(s) from {state_path}")
    return classifier


def classifier_status_lines(classifier: TapClassifier) -> list[str]:
    """Format per-label rejection calibration for the interactive status command."""
    lines: list[str] = []
    for label in sorted(set(classifier.y_labels)):
        calibration = classifier.threshold_calibrations.get(label)
        source = "fallback" if calibration is None else calibration.source
        lines.append(
            f"    {label}: distance={classifier.threshold_for_label(label):.4f} ({source}) "
            f"margin={classifier.margin_threshold:.4f} "
            f"samples={classifier.class_capture_counts()[label]}"
        )
    return lines


def run_application(arguments: argparse.Namespace) -> int:
    """Start capture and serve the interactive command loop."""
    gain_db_values = resolve_gain_db(arguments)
    dispatcher = load_connector_dispatcher(arguments)
    checkpoint_path = resolve_checkpoint_path(arguments.model, arguments.checkpoint)
    processor = SignalProcessor(
        model=arguments.model,
        checkpoint_path=checkpoint_path,
        verify_checkpoint=not arguments.skip_checkpoint_verification,
    )
    checkpoint_sha256 = None
    if arguments.model in {"beats", "pann"}:
        checkpoint_sha256 = processor.prepare()

    state_path = (
        None if arguments.no_state else resolve_state_path(arguments.model, arguments.state_path)
    )
    distance_mode: DistanceMode = (
        "cosine" if arguments.model in {"beats", "pann"} else "standardized_euclidean"
    )
    threshold_mode: ThresholdMode = arguments.threshold_mode or (
        "auto" if arguments.model == "pann" else "fixed"
    )
    threshold = arguments.distance_threshold
    if threshold is None:
        threshold = DEFAULT_DISTANCE_THRESHOLDS[arguments.model]
    metadata = {
        "feature_extractor": arguments.model,
        "checkpoint_sha256": checkpoint_sha256,
        "sample_rate": processor.sample_rate,
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
    classifier = load_or_create_classifier(
        state_path=state_path,
        model_name=arguments.model,
        extractor_metadata=metadata,
        n_neighbors=arguments.neighbors,
        distance_threshold=threshold,
        distance_mode=distance_mode,
        prototype_weight=arguments.prototype_weight,
        spread_multiplier=arguments.spread_multiplier,
        margin_threshold=arguments.margin_threshold,
        probability_temperature=arguments.probability_temperature,
        threshold_mode=threshold_mode,
    )
    print(
        f"[*] Classifier distance: {classifier.distance_mode} "
        f"(threshold mode: {classifier.threshold_mode}, "
        f"fallback/cap: {classifier.distance_threshold:g})"
    )
    print(
        f"[*] Prototype scoring: weight={classifier.prototype_weight:g} "
        f"spread={classifier.spread_multiplier:g} "
        f"margin={classifier.margin_threshold:g} "
        f"temperature={classifier.probability_temperature:g}"
    )
    controller = SwitchController(
        classifier,
        state_path=state_path,
        state_metadata=metadata,
        confirmation_count=arguments.confirmation_count,
        event_cooldown=arguments.event_cooldown,
    )
    print(
        f"[*] Event confirmation: {arguments.confirmation_count} matching capture(s), "
        f"{arguments.event_cooldown:g}s cooldown"
    )
    monitor = None
    if not arguments.no_monitor:
        monitor = MicrophoneMonitor(
            threshold=arguments.energy_threshold,
            sample_rate=processor.sample_rate,
            port=arguments.monitor_port,
        )
        monitor.publish_calibration(controller.calibration_snapshot())
        monitor.start(open_browser=True)
    listener = AudioListener(
        processor,
        device_index=arguments.device_index,
        threshold=arguments.energy_threshold,
        frame_callback=None if monitor is None else monitor.publish,
    )
    dispatcher.start()

    def report_worker_error(error: Exception) -> None:
        print(f"\n[!] Audio processing stopped: {type(error).__name__}: {error}")

    def process_capture(features: np.ndarray, tap_count: int) -> None:
        decision = controller.process_capture(features, tap_count)
        dispatch_decision(decision, dispatcher)
        if monitor is not None:
            monitor.publish_decision(decision)
            monitor.publish_calibration(controller.calibration_snapshot())

    def process_augmented_capture(
        features: np.ndarray,
        tap_count: int,
        augmented_features: tuple[np.ndarray, ...],
    ) -> None:
        decision = controller.process_capture(features, tap_count, augmented_features)
        dispatch_decision(decision, dispatcher)
        if monitor is not None:
            monitor.publish_decision(decision)
            monitor.publish_calibration(controller.calibration_snapshot())

    def listen() -> None:
        try:
            listener.listen(
                process_capture,
                report_worker_error,
                process_augmented_capture,
                controller.is_registering,
                gain_db_values,
                None if monitor is None else monitor.publish_feature_window,
            )
        except Exception as error:
            listener.last_error = error
            report_worker_error(error)

    audio_thread = threading.Thread(
        target=listen,
        name="audio-listener",
        daemon=True,
    )
    audio_thread.start()

    print("\n--- Universal Switch Command Line ---")
    print("Commands:")
    print("  register <label> : Collect examples for a tap surface/location")
    print("  collect-unknown [count] : Collect ordinary sounds (default: 10)")
    print("  clear-unknown    : Delete collected unknown calibration sounds")
    print("  calibrate        : Select and validate classifier parameters")
    print("  cancel           : Cancel the active collection")
    print("  delete <label>   : Delete a class and all of its registered examples")
    print("  list             : List registered labels and example counts")
    print("  status           : Show listener and registration state")
    print("  quit             : Exit the application")
    print("-------------------------------------\n")

    try:
        while audio_thread.is_alive():
            user_input = input(">> ").strip()
            command = user_input.lower()
            if command == "quit":
                break
            if command == "status":
                calibration = controller.calibration_snapshot()
                print(
                    f"[*] mode={controller.mode} exemplars={len(controller.classifier.X_data)} "
                    f"unknowns={len(controller.classifier.negative_data)} "
                    f"calibration={calibration['state']} "
                    f"listener_alive={audio_thread.is_alive()}"
                )
                print(f"[*] {calibration['message']}")
                for line in classifier_status_lines(controller.classifier):
                    print(line)
                connector_status = dispatcher.snapshot()
                if not connector_status:
                    print("[*] connectors=disabled")
                for state in connector_status:
                    delivery = state["last_delivery"]
                    delivery_text = "none"
                    if delivery is not None:
                        outcome = "ok" if delivery["success"] else "failed"
                        delivery_text = f"{outcome}: {delivery['message']}"
                    print(
                        f"[*] connector={state['id']} available={state['available']} "
                        f"queued={state['queue_depth']} last_delivery={delivery_text}"
                    )
            elif command == "list":
                exemplar_counts = controller.classifier.class_counts()
                capture_counts = controller.classifier.class_capture_counts()
                if not exemplar_counts:
                    print("[*] No registered labels.")
                else:
                    print("[*] Registered labels:")
                    for label, count in exemplar_counts.items():
                        captures = capture_counts[label]
                        print(f"    - {label}: {captures} physical capture(s), {count} exemplar(s)")
                print(
                    f"[*] Unknown calibration captures: {len(controller.classifier.negative_data)}"
                )
            elif command.startswith("register "):
                label = user_input.split(" ", 1)[1]
                controller.set_registration_mode(label, taps=arguments.registration_samples)
            elif command == "collect-unknown" or command.startswith("collect-unknown "):
                parts = user_input.split()
                if len(parts) > 2:
                    print("[!] Usage: collect-unknown [count]")
                    continue
                try:
                    count = 10 if len(parts) == 1 else int(parts[1])
                    controller.set_unknown_collection_mode(count)
                except ValueError as error:
                    print(f"[!] {error}")
            elif command == "clear-unknown":
                controller.clear_unknowns()
                if monitor is not None:
                    monitor.publish_calibration(controller.calibration_snapshot())
            elif command == "cancel":
                if not controller.cancel_collection():
                    print("[*] No collection is active.")
            elif command == "calibrate":
                try:
                    controller.calibrate(None if monitor is None else monitor.publish_calibration)
                except Exception as error:
                    print(f"[!] Cannot calibrate: {error}")
            elif command.startswith("delete "):
                label = user_input.split(" ", 1)[1]
                try:
                    controller.delete_class(label)
                except ValueError as error:
                    print(f"[!] {error}")
                if monitor is not None:
                    monitor.publish_calibration(controller.calibration_snapshot())
            else:
                print(
                    "[!] Unknown command. Use 'register <label>', 'collect-unknown [count]', "
                    "'clear-unknown', 'calibrate', 'cancel', 'delete <label>', "
                    "'list', 'status', or 'quit'."
                )
    except (EOFError, KeyboardInterrupt):
        print("\n[*] Shutting down...")
    finally:
        listener.stop()
        audio_thread.join(timeout=3.0)
        dispatcher.close(drain_timeout=3.0)
        if monitor is not None:
            monitor.stop()

    if listener.last_error is not None:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--model", choices=("beats", "pann", "legacy"), default="beats")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--skip-checkpoint-verification", action="store_true")
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--no-state", action="store_true")
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--distance-threshold", type=float)
    parser.add_argument(
        "--threshold-mode",
        choices=("auto", "fixed"),
        help="exemplar rejection thresholds: auto for PANN, fixed for BEATs/legacy by default",
    )
    parser.add_argument("--prototype-weight", type=float, default=0.5)
    parser.add_argument("--spread-multiplier", type=float, default=2.0)
    parser.add_argument("--margin-threshold", type=float)
    parser.add_argument("--probability-temperature", type=float)
    parser.add_argument("--confirmation-count", type=int, default=1)
    parser.add_argument("--event-cooldown", type=float, default=2.0)
    parser.add_argument("--energy-threshold", type=float, default=0.015)
    parser.add_argument("--registration-samples", type=int, default=10)
    gain_group = parser.add_mutually_exclusive_group()
    gain_group.add_argument(
        "--gain-db",
        action="append",
        type=gain_db_argument,
        help=(
            "gain in dB to add per PANN/legacy registration; repeat for multiple "
            "values (unsupported for BEATs)"
        ),
    )
    gain_group.add_argument("--no-gain-augmentation", action="store_true")
    parser.add_argument("--monitor-port", type=int, default=8765)
    parser.add_argument("--no-monitor", action="store_true")
    connector_group = parser.add_mutually_exclusive_group()
    connector_group.add_argument(
        "--connector-config",
        type=Path,
        help=(
            "connector JSON path; defaults to config/connectors.json when that file exists"
        ),
    )
    connector_group.add_argument("--no-connectors", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_arguments = parse_args()
    if cli_arguments.list_devices:
        list_input_devices()
        raise SystemExit(0)
    raise SystemExit(run_application(cli_arguments))
