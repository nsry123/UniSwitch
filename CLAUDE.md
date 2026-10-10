# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python 3.11 only. All commands must run inside the `univ-switch` Conda environment — never install packages globally or create a separate virtualenv.

```bash
conda env create -f environment.yml        # first time
conda env update -n univ-switch -f environment.yml --prune  # update
```

## Commands

```bash
# Syntax check
conda run -n univ-switch python -m compileall main.py src scripts tests

# Tests
conda run -n univ-switch python -m pytest
conda run -n univ-switch python -m pytest tests/test_classifier.py  # single file

# Lint / type-check
conda run -n univ-switch python -m ruff check .
conda run -n univ-switch python -m mypy main.py src scripts

# Run
conda run -n univ-switch python main.py --list-devices
conda run -n univ-switch python main.py --device-index 0

# Offline evaluation
conda run -n univ-switch python -m scripts.evaluate --manifest datasets/manifest.csv
```

BEATs and PANN integration tests skip automatically when their checkpoints are absent. Tests never download models, use a microphone, or call external services.

## Architecture

The system is a local, real-time acoustic knock classifier with no cloud dependency.

**Data flow:** Microphone → `AudioListener` (capture + onset detection) → `SignalProcessor` (feature extraction) → `SwitchController` (registration or inference) → terminal action + `MicrophoneMonitor` (browser dashboard).

**`main.py`** — CLI entry point: parses args, constructs all components, starts the audio thread, and runs the interactive command loop (`register`, `calibrate`, `quit`, etc.).

**`src/audio/`**
- `listener.py` — `AudioListener`: streams PyAudio frames, detects energy onsets, captures fixed-length windows, applies gain augmentation during registration, dispatches to callbacks.
- `new_processor.py` (imported via `processor.py`) — `SignalProcessor`: wraps BEATs (default, 768-D embeddings), PANN, or legacy feature extractor. BEATs uses mean-pooled, L2-normalized embeddings with cosine distance.
- `augmentation.py` — peak-safe gain augmentation applied only during registration, never during inference.
- `beats/` — vendored Microsoft BEATs encoder pinned to a specific UNILM revision; excluded from Ruff and Mypy.

**`src/ml/`**
- `classifier.py` — `TapClassifier`: incremental prototype/exemplar classifier with open-set rejection. Tracks physical capture IDs (for leave-one-out grouping), prototype + exemplar blend scoring, per-class acceptance radii, margin checking, and `.npz` persistence. State schema v4; v1–v3 are intentionally rejected.
- `evaluation.py` — `calibrate_grouped_classifier`: grid-searches hyperparameters using group-aware cross-validation (subject/session/device define leakage groups). Applies an 80% quality gate on known accuracy and unknown rejection before accepting parameters.
- `dataset.py` / `feature_extraction.py` — manifest loading and batch feature extraction for offline evaluation.

**`src/controller.py`** — `SwitchController`: thread-safe orchestration. Maintains mode (`execute` / `register` / `collect_unknown`), confirmation counting, event cooldown, and calibration state. Persists `.npz` state after every capture and calibration.

**`src/web/app.py`** — `MicrophoneMonitor`: a stdlib `ThreadingHTTPServer` serving a single-page dashboard. Exposes `/api/audio` (JSON polling) for waveform, RMS, decision, and calibration data. No WebSocket — the frontend polls.

**`src/bridge.py`** — placeholder for a future switch/action bridge; currently prints to terminal.

**`scripts/`**
- `evaluate.py` — manifest-based offline evaluation with calibration output.
- `install_beats.py` — SHA-256-verified atomic install of the BEATs checkpoint into `models/`.
- `download_pann.py` — downloads and verifies the PANN checkpoint.

## Model / State Details

Each feature extractor has an independent default state file; switching models never silently reuses incompatible registrations:
- BEATs: `state/beats-registrations.npz`
- PANN: `state/registrations.npz`
- Legacy: `state/legacy-registrations.npz`

Default distance thresholds: BEATs `0.15`, PANN `0.20`, legacy `2.5`. Cosine distance for neural models, standardized Euclidean for legacy.

Model checkpoints go in `models/` (gitignored) and must never be committed.

## Coding Conventions

- Keep audio logic in `src/audio/`, ML logic in `src/ml/`, orchestration in `src/controller.py`.
- `src/audio/beats/` is excluded from Ruff (`extend-exclude`) and Mypy — do not add lint directives there.
- Public interfaces require type hints. Docstrings are warranted where thread ownership, array shape, sample rate, or model assumptions are non-obvious.
- Line length: 100. Target: Python 3.11. Ruff rules: `E`, `F`, `I`, `UP`, `B`.
- Use `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for module-level constants.
