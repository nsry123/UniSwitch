# Repository Guidelines

## Project Structure & Module Organization

`main.py` is the command-line entry point: it lists audio devices, starts microphone capture, and exposes the `register <label>` workflow. Core code lives under `src/`: `audio/` handles capture, onset detection, and feature extraction; `ml/` contains the incremental K-nearest-neighbor classifier; and `controller.py` coordinates registration and prediction. `notebooks/audio_test.ipynb` is for audio experiments. `src/bridge.py`, `src/ml/feature_extraction.py`, and `src/web/app.py` are currently placeholders. Large checkpoints belong in the ignored `models/` directory and must not be committed.

## Build, Test, and Development Commands

Work from the repository root and use the existing Conda environment for every Python command:

```bash
conda activate univ-switch
python main.py
```

Do not install packages globally or create a separate virtual environment. Add missing dependencies inside `univ-switch` (for example, `conda run -n univ-switch python -m pip install <package>`), and record project dependencies in `requirements.txt`. `python main.py` starts the interactive microphone listener; enter `register desk` to collect samples and `quit` to stop. The default PANN path expects `models/Cnn14_mAP=0.431.pth`. Run `conda run -n univ-switch python -m compileall main.py src` for a quick syntax check.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for modules, functions, methods, and variables; use `PascalCase` for classes such as `AudioListener` and `TapClassifier`; use uppercase names for true constants. Add type hints to public interfaces and concise docstrings where thread ownership, array shape, sample rate, or model assumptions matter. Keep capture logic in `src/audio`, classification logic in `src/ml`, and orchestration in the controller. No formatter or linter is configured; keep imports grouped and avoid unrelated reformatting.

## Testing Guidelines

There is no automated test suite or coverage threshold yet. Add tests under `tests/` using `pytest`, name files `test_<module>.py`, and name cases `test_<behavior>()`. Prefer deterministic NumPy arrays or small fixtures over live microphone input. When changing audio processing, verify tap counts, feature-vector shape, and classifier rejection behavior. Always run tests through Conda: `conda run -n univ-switch python -m pytest`.

## Commit & Pull Request Guidelines

Keep each commit focused. Pull requests should explain the user-visible behavior, list manual or automated checks, and note audio hardware, sample rate, and checkpoint assumptions. Link relevant issues and include screenshots only for notebook or future web-interface changes. Never commit recordings, secrets, generated outputs, virtual environments, or model weights.
