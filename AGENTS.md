# Repository Guidelines

## Project Structure & Module Organization

`main.py` is the command-line entry point: it lists audio devices, starts microphone capture, and exposes the `register <label>` workflow. Core code lives under `src/`: `audio/` handles capture, onset detection, and feature extraction; `ml/` contains the incremental K-nearest-neighbor classifier plus dataset/evaluation helpers; and `controller.py` coordinates registration and prediction. `scripts/evaluate.py` runs manifest-based offline evaluation, while `scripts/download_pann.py` explicitly downloads and verifies the optional checkpoint. `notebooks/audio_test.ipynb` imports production code for experiments. `src/bridge.py` and `src/web/app.py` document future placeholders. Large checkpoints belong in the ignored `models/` directory and must not be committed.

## Build, Test, and Development Commands

Work from the repository root and use the existing Conda environment for every Python command:

```bash
conda env update -n univ-switch -f environment.yml --prune
conda activate univ-switch
python main.py --list-devices
python main.py --device-index 0
```

Do not install packages globally or create a separate virtual environment. Record runtime dependencies in `requirements.txt`, development dependencies in `requirements-dev.txt`, and Conda/system dependencies in `environment.yml`. `python main.py` starts the interactive microphone listener; enter `register desk` to collect capture-window examples and `quit` to stop. The default PANN path expects `models/Cnn14_mAP=0.431.pth`; obtain it explicitly with `python scripts/download_pann.py`. Run `conda run -n univ-switch python -m compileall main.py src scripts tests` for a quick syntax check.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for modules, functions, methods, and variables; use `PascalCase` for classes such as `AudioListener` and `TapClassifier`; use uppercase names for true constants. Add type hints to public interfaces and concise docstrings where thread ownership, array shape, sample rate, or model assumptions matter. Keep capture logic in `src/audio`, classification logic in `src/ml`, and orchestration in the controller. Use Ruff for formatting/linting and Mypy for type checking; avoid unrelated reformatting.

## Testing Guidelines

Add tests under `tests/` using `pytest`, name files `test_<module>.py`, and name cases `test_<behavior>()`. Prefer deterministic NumPy arrays or small PCM WAV fixtures over live microphone input. When changing audio processing, verify tap counts, feature-vector shape, classifier rejection behavior, persistence compatibility, and group leakage checks. The PANN integration test must skip when the ignored checkpoint is absent and must never download it. Always run checks through Conda: `conda run -n univ-switch python -m pytest`, `conda run -n univ-switch python -m ruff check .`, and `conda run -n univ-switch python -m mypy main.py src scripts`.

## Commit & Pull Request Guidelines

Keep each commit focused. Pull requests should explain the user-visible behavior, list manual or automated checks, and note audio hardware, sample rate, and checkpoint assumptions. Link relevant issues and include screenshots only for notebook or future web-interface changes. Never commit recordings, secrets, generated outputs, virtual environments, or model weights.
