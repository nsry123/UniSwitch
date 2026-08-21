# Universal Switch

Universal Switch is a local prototype that learns user-defined labels for the
surface or location of acoustic knocks, such as `desk` or `wall`. It also counts
threshold-qualified knocks within each captured window. By default, a frozen
Microsoft BEATs Iter3+ AS2M encoder creates 768-D audio embeddings and the local
prototype/exemplar classifier learns the user's labels. No cloud service is involved.

The switch/action bridge is not implemented. A successful classification
currently prints an action message to the terminal.

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

## BEATs checkpoint

The default application requires the official pretraining-only
`BEATs_iter3+ (AS2M)` checkpoint from the
[Microsoft BEATs release](https://github.com/microsoft/unilm/tree/master/beats).
The model is not committed and is never downloaded automatically. Obtain the
official file and its SHA-256 digest through a trusted channel, then install it:

```bash
conda run -n univ-switch python scripts/install_beats.py \
  /path/to/official/BEATs_iter3_plus_AS2M.pt \
  --sha256 <trusted-64-character-digest>
```

The installer verifies the source before atomically writing
`models/BEATs_iter3_plus_AS2M.pt` and its `.sha256` sidecar. Runtime verification
uses that sidecar and strict checkpoint loading. The vendored encoder source is
pinned to Microsoft UNILM revision
`833df7e7832e5064a281131ee64a481afa8e5b95`.

PANN remains available as an alternative. Its official checkpoint downloader is
unchanged:

```bash
conda run -n univ-switch python scripts/download_pann.py
conda run -n univ-switch python main.py --model pann --device-index 0
```

## Run and register labels

List devices without starting capture:

```bash
conda run -n univ-switch python main.py --list-devices
```

Start the listener, selecting the correct input device index:

```bash
conda run -n univ-switch python main.py --device-index 0
```

The application opens a localhost microphone monitor in your browser. Its live
charts show the latest waveform, RMS energy, and the labeled trigger threshold,
making it easier to tune `--energy-threshold`. Use `--monitor-port 9000` to
select another local port or `--no-monitor` to run without the dashboard. The
same energy threshold is used to count knocks inside each capture. A counted
knock must fall below 70% of the threshold before another knock can register,
which prevents table ringing and threshold chatter from inflating the count.

At the prompt, enter `register desk`. Each threshold-triggered capture window is
one registration example, even if energy detection finds multiple knocks inside
that window. BEATs registrations are saved to
`state/beats-registrations.npz` by default and loaded on the next startup. Enter
`delete desk` to remove that class and all of
its examples; deleting the final class also removes the empty state file. Enter
`list` to show every registered label and its example count. Enter `quit` or
press Ctrl-C for a graceful shutdown.
Mean-pooled BEATs embeddings use L2 normalization and cosine distance with a
provisional global ceiling of `0.15`; prefer the offline evaluator's calibrated
value when representative validation data is available. Each class becomes
eligible after three examples and is represented by its prototype, registered
exemplars, and leave-one-out spread. A prediction must pass both its
class-specific distance radius and the margin from the runner-up class. Known
class probabilities are derived from the blended prototype/exemplar scores.

An accepted capture triggers its action immediately. After an event, matching
captures are suppressed for two seconds. Rejected captures clear pending
evidence. Optional multi-capture confirmation is available through
`--confirmation-count`; tune cooldown with `--event-cooldown`. Scoring controls are
available through `--prototype-weight`, `--spread-multiplier`,
`--margin-threshold`, and `--probability-temperature`. Persisted state files keep
their scoring configuration. Schema-v2 registrations are upgraded in memory and
written as schema v3 on the next save.

Each feature extractor has an independent default state file, so switching
models never silently reuses incompatible registrations:

- BEATs: `state/beats-registrations.npz`
- PANN: `state/registrations.npz`
- Legacy: `state/legacy-registrations.npz`

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
  --save-model outputs/evaluated_classifier.npz
```

The evaluator calibrates exemplar count, rejection distance, prototype weight,
spread multiplier, ambiguity margin, and probability temperature on the
validation split. It then reports known-label classification, unknown rejection,
and known-class probability negative log-likelihood on the test split. A label
absent from the training split is treated as unknown. These metrics describe only
the supplied manifest; the repository does not ship a dataset or claim model
quality.

## Development checks

```bash
conda run -n univ-switch python -m compileall main.py src scripts tests
conda run -n univ-switch python -m pytest
conda run -n univ-switch python -m ruff check .
conda run -n univ-switch python -m mypy main.py src scripts
```

The BEATs and PANN integration tests skip when their optional checkpoints are
absent. Tests never download models, use a microphone, or call external services.

## Troubleshooting

- **No input device:** run with `--list-devices`, then pass the desired index.
- **BEATs checkpoint error:** install the official pretraining checkpoint and
  trusted digest with `scripts/install_beats.py`. Do not use an AudioSet
  fine-tuned checkpoint.
- **PANN checkpoint error:** run `scripts/download_pann.py` and select
  `--model pann`, or pass `--checkpoint /trusted/path/checkpoint.pth`.
- **M4A cannot be decoded:** install FFmpeg or recreate the Conda environment.
- **Numba cache is unwritable:** set `NUMBA_CACHE_DIR` to a writable cache path.
- **Worker stopped:** the CLI prints the processing exception and stops capture;
  fix the reported model/audio error before restarting.
- **Monitor port is already in use:** pass another value such as
  `--monitor-port 9000`.
- **Schema-v1 PANN registration error:** re-register using a new `--state-path`
  or move/remove the old state file. Legacy registration files remain compatible.
- **Existing PANN registrations:** start with `--model pann`; BEATs cannot reuse
  embeddings created by another encoder.
