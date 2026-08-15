# Universal Switch

Universal Switch is a local prototype that learns user-defined labels for the
surface or location of acoustic knocks, such as `desk` or `wall`. It also counts
knock onsets within each captured window. A pretrained PANN CNN14 network creates
audio embeddings; an in-memory or persisted scikit-learn KNN model learns the
user's labels. No cloud service is involved.

The switch/action bridge and web UI are not implemented. A successful
classification currently prints an action message to the terminal.

## Setup

Python 3.11 is supported. Use the project Conda environment; FFmpeg is included
so Librosa can decode AAC/M4A recordings and PortAudio supports PyAudio.

```bash
conda env create -f environment.yml
conda activate univ-switch
```

To update an existing environment:

```bash
conda env update -n univ-switch -f environment.yml --prune
```

`requirements.txt` contains runtime dependencies and `requirements-dev.txt`
adds test, lint, and type-check tools. On systems where PyAudio is installed by
pip, PortAudio development headers must already be available.

## PANN checkpoint

The default application requires the official PANN CNN14 checkpoint. It is not
committed because it is approximately 312 MB.

```bash
conda run -n univ-switch python scripts/download_pann.py
```

The downloader writes `models/Cnn14_mAP=0.431.pth` only after verifying SHA-256
`0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31`.
It never runs automatically. You can instead place a trusted copy at that path.

## Run and register labels

List devices without starting capture:

```bash
conda run -n univ-switch python main.py --list-devices
```

Start the listener, selecting the correct input device index:

```bash
conda run -n univ-switch python main.py --device-index 0
```

At the prompt, enter `register desk`. Each threshold-triggered capture window is
one registration example, even if onset detection finds multiple knocks inside
that window. Registrations are saved to `state/registrations.npz` by default and
loaded on the next startup. Enter `quit` or press Ctrl-C for a graceful shutdown.

For the faster 30-value legacy feature extractor:

```bash
conda run -n univ-switch python main.py --model legacy --device-index 0
```

## Dataset and evaluation

Offline evaluation requires a CSV manifest based on
`datasets/manifest.example.csv`. Use PCM WAV recordings for portable tests and
experiments. `subject`, `session`, and `device` collectively define a leakage
group; a group may occur in only one split.

```bash
conda run -n univ-switch python -m scripts.evaluate \
  --manifest datasets/manifest.csv \
  --model legacy \
  --save-model outputs/evaluated_classifier.npz
```

The evaluator calibrates K and the unknown-rejection distance on the validation
split, then reports known-label classification and unknown-rejection metrics on
the test split. A label absent from the training split is treated as unknown.
These metrics describe only the supplied manifest; the repository does not ship
a dataset or claim model quality.

## Development checks

```bash
conda run -n univ-switch python -m compileall main.py src scripts tests
conda run -n univ-switch python -m pytest
conda run -n univ-switch python -m ruff check .
conda run -n univ-switch python -m mypy main.py src scripts
```

The PANN integration test is skipped when the checkpoint is absent. Tests never
download models, use a microphone, or call external services.

## Troubleshooting

- **No input device:** run with `--list-devices`, then pass the desired index.
- **Checkpoint error:** run `scripts/download_pann.py` or pass
  `--checkpoint /trusted/path/checkpoint.pth`.
- **M4A cannot be decoded:** install FFmpeg or recreate the Conda environment.
- **Numba cache is unwritable:** set `NUMBA_CACHE_DIR` to a writable cache path.
- **Worker stopped:** the CLI prints the processing exception and stops capture;
  fix the reported model/audio error before restarting.
