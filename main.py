"""Command-line application for interactive knock registration and inference."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import pyaudio

from src.audio.listener import AudioListener
from src.audio.processor import PANN_CHECKPOINT_PATH, SignalProcessor
from src.controller import SwitchController
from src.ml.classifier import DistanceMode, TapClassifier
from src.web.app import MicrophoneMonitor

DEFAULT_STATE_PATH = Path("state/registrations.npz")


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
    checkpoint_sha256: str | None,
    n_neighbors: int,
    distance_threshold: float,
    distance_mode: DistanceMode,
    prototype_weight: float,
    spread_multiplier: float,
    margin_threshold: float | None,
    probability_temperature: float | None,
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
        )

    classifier = TapClassifier.load(state_path)
    saved_model = classifier.metadata.get("feature_extractor")
    saved_sha256 = classifier.metadata.get("checkpoint_sha256")
    if saved_model != model_name:
        raise ValueError(
            f"state file uses feature extractor {saved_model!r}, not {model_name!r}; "
            "select the matching --model or a different --state-path"
        )
    if model_name == "pann" and saved_sha256 != checkpoint_sha256:
        raise ValueError(
            "state file checkpoint hash differs from the selected PANN checkpoint; "
            "use a compatible state file"
        )
    if classifier.distance_mode != distance_mode:
        raise ValueError(
            f"state file uses distance mode {classifier.distance_mode!r}, "
            f"not {distance_mode!r}; use a compatible state file"
        )
    print(f"[*] Loaded {len(classifier.X_data)} registration(s) from {state_path}")
    return classifier


def run_application(arguments: argparse.Namespace) -> int:
    """Start capture and serve the interactive command loop."""
    processor = SignalProcessor(
        model=arguments.model,
        checkpoint_path=arguments.checkpoint,
        verify_checkpoint=not arguments.skip_checkpoint_verification,
    )
    checkpoint_sha256 = None
    if arguments.model == "pann":
        checkpoint_sha256 = processor.prepare()

    state_path = None if arguments.no_state else arguments.state_path
    distance_mode: DistanceMode = (
        "cosine" if arguments.model == "pann" else "standardized_euclidean"
    )
    threshold = arguments.distance_threshold
    if threshold is None:
        threshold = 0.20 if arguments.model == "pann" else 2.5
    classifier = load_or_create_classifier(
        state_path=state_path,
        model_name=arguments.model,
        checkpoint_sha256=checkpoint_sha256,
        n_neighbors=arguments.neighbors,
        distance_threshold=threshold,
        distance_mode=distance_mode,
        prototype_weight=arguments.prototype_weight,
        spread_multiplier=arguments.spread_multiplier,
        margin_threshold=arguments.margin_threshold,
        probability_temperature=arguments.probability_temperature,
    )
    print(
        f"[*] Classifier distance: {classifier.distance_mode} "
        f"(rejection threshold: {classifier.distance_threshold:g})"
    )
    print(
        f"[*] Prototype scoring: weight={classifier.prototype_weight:g} "
        f"spread={classifier.spread_multiplier:g} "
        f"margin={classifier.margin_threshold:g} "
        f"temperature={classifier.probability_temperature:g}"
    )
    metadata = {
        "feature_extractor": arguments.model,
        "checkpoint_sha256": checkpoint_sha256,
        "sample_rate": processor.sample_rate,
    }
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
        monitor.start(open_browser=True)
    listener = AudioListener(
        processor,
        device_index=arguments.device_index,
        threshold=arguments.energy_threshold,
        frame_callback=None if monitor is None else monitor.publish,
    )

    def report_worker_error(error: Exception) -> None:
        print(f"\n[!] Audio processing stopped: {type(error).__name__}: {error}")

    def listen() -> None:
        try:
            listener.listen(controller.audio_callback, report_worker_error)
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
                print(
                    f"[*] mode={controller.mode} registrations={len(classifier.X_data)} "
                    f"listener_alive={audio_thread.is_alive()}"
                )
            elif command.startswith("register "):
                label = user_input.split(" ", 1)[1]
                controller.set_registration_mode(label, taps=arguments.registration_samples)
            else:
                print("[!] Unknown command. Use 'register <label>', 'status', or 'quit'.")
    except (EOFError, KeyboardInterrupt):
        print("\n[*] Shutting down...")
    finally:
        listener.stop()
        audio_thread.join(timeout=3.0)
        if monitor is not None:
            monitor.stop()

    if listener.last_error is not None:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--model", choices=("pann", "legacy"), default="pann")
    parser.add_argument("--checkpoint", type=Path, default=PANN_CHECKPOINT_PATH)
    parser.add_argument("--skip-checkpoint-verification", action="store_true")
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--no-state", action="store_true")
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--distance-threshold", type=float)
    parser.add_argument("--prototype-weight", type=float, default=0.5)
    parser.add_argument("--spread-multiplier", type=float, default=2.0)
    parser.add_argument("--margin-threshold", type=float)
    parser.add_argument("--probability-temperature", type=float)
    parser.add_argument("--confirmation-count", type=int, default=1)
    parser.add_argument("--event-cooldown", type=float, default=2.0)
    parser.add_argument("--energy-threshold", type=float, default=0.015)
    parser.add_argument("--registration-samples", type=int, default=10)
    parser.add_argument("--monitor-port", type=int, default=8765)
    parser.add_argument("--no-monitor", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_arguments = parse_args()
    if cli_arguments.list_devices:
        list_input_devices()
        raise SystemExit(0)
    raise SystemExit(run_application(cli_arguments))
