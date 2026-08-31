from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Final

import librosa
import numpy as np

PANN_CHECKPOINT_PATH: Final = Path("models/Cnn14_mAP=0.431.pth")
PANN_CHECKPOINT_SHA256: Final = "0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31"
BEATS_CHECKPOINT_PATH: Final = Path("models/BEATs_iter3_plus_AS2M.pt")
BEATS_SOURCE_REVISION: Final = "833df7e7832e5064a281131ee64a481afa8e5b95"
BEATS_EMBEDDING_DIM: Final = 768
BEATS_POOLING: Final = "mean"
BEATS_FEATURE_ALIGNMENT: Final = "first_threshold_onset"
BEATS_FEATURE_WINDOW_SECONDS: Final = 0.580
BEATS_PRE_ONSET_SECONDS: Final = 0.200
DEFAULT_ENERGY_THRESHOLD: Final = 0.015
KNOCK_RMS_FRAME_SECONDS: Final = 0.020
KNOCK_RMS_HOP_SECONDS: Final = 0.010
KNOCK_RELEASE_RATIO: Final = 0.7
KNOCK_RELEASE_SECONDS: Final = 0.060
MIN_KNOCK_INTERVAL_SECONDS: Final = 0.120


class SignalProcessor:
    """
    Handles transient detection (counting taps) and feature extraction
    (generating a sonic fingerprint) for incoming audio buffers.
    """

    def __init__(
        self,
        model: str = "beats",
        sample_rate: int = 16000,
        hop_length: int = 512,
        checkpoint_path: str | Path | None = None,
        verify_checkpoint: bool = True,
    ) -> None:
        normalized_model = model.lower()
        if normalized_model == "default":
            normalized_model = "beats"
        if normalized_model not in {"beats", "legacy", "pann"}:
            raise ValueError("model must be one of: beats, legacy, pann")
        if sample_rate <= 0 or hop_length <= 0:
            raise ValueError("sample_rate and hop_length must be positive")

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.model = normalized_model
        default_checkpoint = {
            "beats": BEATS_CHECKPOINT_PATH,
            "pann": PANN_CHECKPOINT_PATH,
        }.get(normalized_model)
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else default_checkpoint
        )
        self.verify_checkpoint = verify_checkpoint
        self.checkpoint_sha256: str | None = None
        if self.model == "beats":
            self.sample_rate = 16000
        elif self.model == "pann":
            self.sample_rate = 32000

        # Neural encoders are loaded lazily. prepare() loads the selected encoder
        # before capture so checkpoint errors are reported at startup.
        self._pann_model: Any | None = None
        self._pann_device: Any | None = None
        self._beats_model: Any | None = None
        self._beats_device: Any | None = None

    def prepare(self) -> str | None:
        """Load resources required for feature extraction before capture starts."""
        if self.model == "legacy":
            return None
        assert self.checkpoint_path is not None
        if self.model == "beats" and self._beats_model is None:
            self._load_beats(str(self.checkpoint_path))
        elif self.model == "pann" and self._pann_model is None:
            self._load_pann(str(self.checkpoint_path))
        return self.checkpoint_sha256

    # ------------------------------------------------------------------ #
    #  Energy gate (used by the listener thread)                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_audio(audio_clip: np.ndarray) -> np.ndarray:
        """Validate and normalize a waveform passed to feature extraction."""
        waveform = np.asarray(audio_clip)
        if waveform.ndim != 1:
            raise ValueError(f"audio_clip must be one-dimensional, got shape {waveform.shape}")
        if waveform.size == 0:
            raise ValueError("audio_clip must not be empty")
        if not np.issubdtype(waveform.dtype, np.number):
            raise TypeError("audio_clip must contain numeric samples")
        if not np.isfinite(waveform).all():
            raise ValueError("audio_clip contains NaN or infinite samples")
        return waveform.astype(np.float32, copy=False)

    def calculate_energy(self, audio_frame: np.ndarray) -> float:
        """
        Calculates the Root Mean Square (RMS) energy of a raw audio frame.
        Used by the listener thread to detect if a sound broke the noise floor.
        """
        waveform = np.asarray(audio_frame)
        if waveform.ndim != 1:
            raise ValueError("audio_frame must be one-dimensional")
        if len(waveform) == 0:
            return 0.0
        if not np.isfinite(waveform).all():
            raise ValueError("audio_frame contains NaN or infinite samples")
        return float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))

    def prepare_feature_audio(
        self,
        audio_clip: np.ndarray,
        trigger_sample: int,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    ) -> np.ndarray:
        """Return audio shaped for the selected feature extractor.

        BEATs receives a fixed first-impact window so its temporal mean pool is
        not dominated by the remainder of the capture. ``trigger_sample`` is
        the coarse start of the stream chunk that opened the live energy gate;
        the onset is refined from there with the same RMS framing used for tap
        counting. Other extractors retain their historical full-clip input.
        """
        waveform = self._validate_audio(audio_clip)
        bounds = self.feature_window_bounds(waveform, trigger_sample, energy_threshold)
        if bounds is None:
            return waveform
        source_start, _, source_end = bounds

        window_samples = round(BEATS_FEATURE_WINDOW_SECONDS * self.sample_rate)
        copy_start = max(0, source_start)
        copy_end = min(waveform.size, source_end)
        destination_start = copy_start - source_start

        prepared = np.zeros(window_samples, dtype=np.float32)
        prepared[destination_start : destination_start + copy_end - copy_start] = waveform[
            copy_start:copy_end
        ]
        return prepared

    def feature_window_bounds(
        self,
        audio_clip: np.ndarray,
        trigger_sample: int,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    ) -> tuple[int, int, int] | None:
        """Return nominal ``(start, onset, end)`` samples for feature input.

        Bounds may extend beyond the waveform where zero-padding is part of the
        BEATs input. Other extractors return ``None`` because they consume the
        complete capture.
        """
        waveform = self._validate_audio(audio_clip)
        if self.model != "beats":
            return None
        if not isinstance(trigger_sample, (int, np.integer)):
            raise TypeError("trigger_sample must be an integer sample index")
        if trigger_sample < 0 or trigger_sample >= waveform.size:
            raise ValueError(
                f"trigger_sample must be within the waveform, got {trigger_sample}"
            )
        if not np.isfinite(energy_threshold) or energy_threshold < 0:
            raise ValueError("energy_threshold must be finite and non-negative")

        frame_length = max(1, round(KNOCK_RMS_FRAME_SECONDS * self.sample_rate))
        hop_length = max(1, round(KNOCK_RMS_HOP_SECONDS * self.sample_rate))
        search_waveform = waveform[int(trigger_sample) :]
        energy = self._rms_envelope(search_waveform, frame_length, hop_length)
        crossings = np.flatnonzero(energy > energy_threshold)
        if crossings.size == 0:
            raise ValueError(
                "no threshold-qualified onset found at or after trigger_sample; "
                "check the capture or energy threshold"
            )

        onset_sample = int(trigger_sample) + int(crossings[0]) * hop_length
        window_samples = round(BEATS_FEATURE_WINDOW_SECONDS * self.sample_rate)
        pre_onset_samples = round(BEATS_PRE_ONSET_SECONDS * self.sample_rate)
        source_start = onset_sample - pre_onset_samples
        source_end = source_start + window_samples
        return source_start, onset_sample, source_end

    # ------------------------------------------------------------------ #
    #  Knock counting                                                      #
    # ------------------------------------------------------------------ #

    def count_taps(
        self,
        audio_clip: np.ndarray,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    ) -> int:
        """Count threshold-crossing impacts in a raw mono waveform.

        The high threshold is shared with live capture. After an impact, the
        detector requires both a quiet release period below 70% of the high
        threshold and a minimum onset interval before it can count again.
        """
        waveform = self._validate_audio(audio_clip)
        if not np.isfinite(energy_threshold) or energy_threshold < 0:
            raise ValueError("energy_threshold must be finite and non-negative")

        frame_length = max(1, round(KNOCK_RMS_FRAME_SECONDS * self.sample_rate))
        hop_length = max(1, round(KNOCK_RMS_HOP_SECONDS * self.sample_rate))
        release_frames = max(1, round(KNOCK_RELEASE_SECONDS / KNOCK_RMS_HOP_SECONDS))
        minimum_interval_frames = max(
            1, round(MIN_KNOCK_INTERVAL_SECONDS / KNOCK_RMS_HOP_SECONDS)
        )
        energy = self._rms_envelope(waveform, frame_length, hop_length)

        release_threshold = KNOCK_RELEASE_RATIO * energy_threshold
        armed = True
        quiet_frames = 0
        last_knock_frame: int | None = None
        knock_count = 0

        for frame_index, frame_energy in enumerate(energy):
            if armed:
                if frame_energy > energy_threshold:
                    knock_count += 1
                    last_knock_frame = frame_index
                    armed = False
                    quiet_frames = 0
                continue

            if frame_energy <= release_threshold:
                quiet_frames += 1
            else:
                quiet_frames = 0

            interval_elapsed = (
                last_knock_frame is not None
                and frame_index - last_knock_frame >= minimum_interval_frames
            )
            if quiet_frames >= release_frames and interval_elapsed:
                armed = True

        return knock_count

    @staticmethod
    def _rms_envelope(
        waveform: np.ndarray, frame_length: int, hop_length: int
    ) -> np.ndarray:
        """Return non-centered RMS frames without normalizing their amplitude."""
        if waveform.size < frame_length:
            return np.array(
                [np.sqrt(np.mean(waveform.astype(np.float64) ** 2))]
            )

        squared = waveform.astype(np.float64) ** 2
        cumulative_energy = np.concatenate(([0.0], np.cumsum(squared)))
        starts = np.arange(0, waveform.size - frame_length + 1, hop_length)
        frame_energy = cumulative_energy[starts + frame_length] - cumulative_energy[starts]
        return np.sqrt(frame_energy / frame_length)

    # ------------------------------------------------------------------ #
    #  Legacy MFCC feature extractor (kept for backwards compatibility)   #
    # ------------------------------------------------------------------ #

    def extract_features(self, audio_clip: np.ndarray) -> np.ndarray:
        waveform = self._validate_audio(audio_clip)
        if self.model == "beats":
            return self.extract_features_beats(audio_clip=waveform)
        if self.model == "pann":
            return self.extract_features_pann(audio_clip=waveform)
        return self.extract_features_default(audio_clip=waveform)

    def extract_features_default(self, audio_clip: np.ndarray) -> np.ndarray:
        """
        Compresses a 1.5-second audio array into a flat, 1D feature vector
        representing the acoustic profile of the tap.
        """
        waveform = self._validate_audio(audio_clip)
        _, percussive = librosa.effects.hpss(waveform)

        mfccs = librosa.feature.mfcc(y=percussive, sr=self.sample_rate, n_mfcc=13)
        centroid = librosa.feature.spectral_centroid(y=percussive, sr=self.sample_rate)
        rolloff = librosa.feature.spectral_rolloff(y=percussive, sr=self.sample_rate)

        feature_vector = np.hstack(
            (
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.mean(centroid),
                np.std(centroid),
                np.mean(rolloff),
                np.std(rolloff),
            )
        )
        return feature_vector.astype(np.float64, copy=False)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def validate_pann_checkpoint(self) -> str:
        """Validate checkpoint presence and, by default, official SHA-256."""
        assert self.checkpoint_path is not None
        resolved_checkpoint = self.checkpoint_path.expanduser().resolve()
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(
                f"PANN checkpoint not found: {resolved_checkpoint}. "
                "Run 'python scripts/download_pann.py' or pass --checkpoint."
            )
        actual_sha256 = self._sha256_file(resolved_checkpoint)
        if self.verify_checkpoint and actual_sha256 != PANN_CHECKPOINT_SHA256:
            raise ValueError(
                "PANN checkpoint checksum mismatch: "
                f"expected {PANN_CHECKPOINT_SHA256}, received {actual_sha256}"
            )
        self.checkpoint_sha256 = actual_sha256
        return actual_sha256

    def validate_beats_checkpoint(self) -> str:
        """Validate a manually installed BEATs checkpoint against its sidecar digest."""
        assert self.checkpoint_path is not None
        resolved_checkpoint = self.checkpoint_path.expanduser().resolve()
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(
                f"BEATs checkpoint not found: {resolved_checkpoint}. "
                "Run 'python scripts/install_beats.py /path/to/official/checkpoint "
                "--sha256 <trusted-digest>'."
            )
        actual_sha256 = self._sha256_file(resolved_checkpoint)
        sidecar = resolved_checkpoint.with_suffix(".sha256")
        if self.verify_checkpoint:
            if not sidecar.is_file():
                raise FileNotFoundError(
                    f"BEATs checksum sidecar not found: {sidecar}. "
                    "Install the checkpoint with scripts/install_beats.py."
                )
            sidecar_tokens = sidecar.read_text(encoding="utf-8").split()
            if not sidecar_tokens:
                raise ValueError(f"invalid BEATs checksum sidecar: {sidecar}")
            expected_sha256 = sidecar_tokens[0].lower()
            if len(expected_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in expected_sha256
            ):
                raise ValueError(f"invalid BEATs checksum sidecar: {sidecar}")
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "BEATs checkpoint checksum mismatch: "
                    f"expected {expected_sha256}, received {actual_sha256}"
                )
        self.checkpoint_sha256 = actual_sha256
        return actual_sha256

    def _load_beats(self, checkpoint_path: str) -> None:
        """Strictly load the pinned BEATs pretraining checkpoint."""
        import torch

        from src.audio.beats import BEATs, BEATsConfig

        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
        original_checkpoint = self.checkpoint_path
        self.checkpoint_path = resolved_checkpoint
        try:
            self.validate_beats_checkpoint()
        finally:
            self.checkpoint_path = original_checkpoint

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        checkpoint = torch.load(resolved_checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not {"cfg", "model"}.issubset(checkpoint):
            raise ValueError("BEATs checkpoint must contain 'cfg' and 'model' mappings")
        if not isinstance(checkpoint["cfg"], dict) or not isinstance(checkpoint["model"], dict):
            raise ValueError("BEATs checkpoint cfg and model entries must be mappings")
        config = BEATsConfig(checkpoint["cfg"])
        if config.finetuned_model:
            raise ValueError("expected a BEATs pretraining checkpoint, not a fine-tuned model")
        if config.encoder_embed_dim != BEATS_EMBEDDING_DIM:
            raise ValueError(
                f"expected a {BEATS_EMBEDDING_DIM}-D BEATs encoder, "
                f"received {config.encoder_embed_dim}"
            )
        model = BEATs(config)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.requires_grad_(False)
        model.eval()
        model.to(device)
        self._beats_model = model
        self._beats_device = device
        print(f"[*] BEATs encoder loaded on {device} from {resolved_checkpoint}")

    def extract_features_beats(self, audio_clip: np.ndarray) -> np.ndarray:
        """Return a mean-pooled 768-D representation from the frozen BEATs encoder."""
        import torch

        waveform_array = self._validate_audio(audio_clip)
        if self._beats_model is None:
            assert self.checkpoint_path is not None
            self._load_beats(str(self.checkpoint_path))
        assert self._beats_model is not None and self._beats_device is not None
        waveform = torch.from_numpy(waveform_array.astype(np.float32)).unsqueeze(0)
        with torch.inference_mode():
            # The official Kaldi-compatible frontend runs on CPU. Only the compact
            # filterbank representation is moved to the selected accelerator.
            fbank = self._beats_model.preprocess(waveform).unsqueeze(1)
            features = self._beats_model.patch_embedding(fbank.to(self._beats_device))
            features = features.reshape(features.shape[0], features.shape[1], -1)
            features = features.transpose(1, 2)
            features = self._beats_model.layer_norm(features)
            if self._beats_model.post_extract_proj is not None:
                features = self._beats_model.post_extract_proj(features)
            features = self._beats_model.dropout_input(features)
            encoded, _ = self._beats_model.encoder(features)
            embedding = encoded.mean(dim=1)
        result = embedding.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        if result.shape != (BEATS_EMBEDDING_DIM,) or not np.isfinite(result).all():
            raise ValueError(
                f"BEATs returned an invalid embedding: shape={result.shape}, "
                f"finite={bool(np.isfinite(result).all())}"
            )
        return result

    # ------------------------------------------------------------------ #
    #  PANN CNN14 feature extractor                                       #
    # ------------------------------------------------------------------ #

    def _load_pann(self, checkpoint_path: str) -> None:
        """
        Builds the CNN14 model, loads the pretrained checkpoint, and moves
        it to the best available device (MPS on Apple Silicon, CUDA on
        Nvidia, CPU otherwise).

        The model and its log-mel spectrogram transform are cached on the
        instance so this is only called once per process lifetime.

        Checkpoint
        ----------
        Download CNN14_mAP=0.431.pth from the PANNS Zenodo release:
        https://zenodo.org/record/3987831

        Dependencies
        ------------
        pip install torch torchlibrosa
        torchlibrosa supplies the differentiable log-mel frontend that matches
        the exact normalisation used during PANN pretraining. Using librosa's
        mel instead would silently produce out-of-distribution inputs and
        degrade embedding quality.
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchlibrosa.stft import LogmelFilterBank, Spectrogram

        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
        original_checkpoint = self.checkpoint_path
        self.checkpoint_path = resolved_checkpoint
        try:
            self.validate_pann_checkpoint()
        except Exception:
            self.checkpoint_path = original_checkpoint
            raise

        # ── Device selection ────────────────────────────────────────────
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # ── Log-mel frontend ────────────────────────────────────────────
        # These hyperparameters are fixed by the PANN pretraining recipe.
        # Do not change them — any deviation produces out-of-distribution
        # inputs and silently degrades the embedding.
        SAMPLE_RATE = 32000  # PANN was pretrained at 32 kHz
        WINDOW_SIZE = 1024
        HOP_SIZE = 320
        MEL_BINS = 64
        FMIN = 50
        FMAX = 14000

        spectrogram_extractor = Spectrogram(
            n_fft=WINDOW_SIZE,
            hop_length=HOP_SIZE,
            win_length=WINDOW_SIZE,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        logmel_extractor = LogmelFilterBank(
            sr=SAMPLE_RATE,
            n_fft=WINDOW_SIZE,
            n_mels=MEL_BINS,
            fmin=FMIN,
            fmax=FMAX,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )

        # ── CNN14 architecture ──────────────────────────────────────────
        # Reproduced from qiuqiangkong/panns_inference (MIT licence).
        # CNN14 stacks six ConvBlock modules (each a pair of 3×3 conv layers
        # with BN and ReLU, followed by average pooling) then collapses the
        # time-frequency map with global average + max pooling into a 2048-d
        # embedding. Module names deliberately match the official checkpoint;
        # strict loading below ensures no part of the backbone stays randomly
        # initialized because of a silent key mismatch.

        def init_layer(layer: nn.Module) -> None:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        def init_bn(bn: nn.BatchNorm2d) -> None:
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)

        class ConvBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.bn2 = nn.BatchNorm2d(out_ch)
                init_layer(self.conv1)
                init_layer(self.conv2)
                init_bn(self.bn1)
                init_bn(self.bn2)

            def forward(self, x, pool_size=(2, 2), pool_type="avg"):
                x = F.relu_(self.bn1(self.conv1(x)))
                x = F.relu_(self.bn2(self.conv2(x)))
                if pool_type == "max":
                    x = F.max_pool2d(x, pool_size)
                elif pool_type == "avg":
                    x = F.avg_pool2d(x, pool_size)
                elif pool_type == "avg+max":
                    x = F.avg_pool2d(x, pool_size) + F.max_pool2d(x, pool_size)
                return x

        class Cnn14(nn.Module):
            def __init__(self):
                super().__init__()
                self.spectrogram_extractor = spectrogram_extractor
                self.logmel_extractor = logmel_extractor
                self.bn0 = nn.BatchNorm2d(64)
                self.conv_block1 = ConvBlock(1, 64)
                self.conv_block2 = ConvBlock(64, 128)
                self.conv_block3 = ConvBlock(128, 256)
                self.conv_block4 = ConvBlock(256, 512)
                self.conv_block5 = ConvBlock(512, 1024)
                self.conv_block6 = ConvBlock(1024, 2048)
                self.fc1 = nn.Linear(2048, 2048, bias=True)
                self.fc_audioset = nn.Linear(2048, 527, bias=True)
                init_bn(self.bn0)
                init_layer(self.fc1)
                init_layer(self.fc_audioset)

            def forward(self, waveform: torch.Tensor):
                x = self.spectrogram_extractor(waveform)  # (B, 1, T, F)
                x = self.logmel_extractor(x)  # (B, 1, T, 64)

                x = x.transpose(1, 3)  # (B, 64, T, 1)
                x = self.bn0(x)
                x = x.transpose(1, 3)  # (B, 1, T, 64)

                x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
                x = F.dropout(x, p=0.2, training=self.training)
                x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
                x = F.dropout(x, p=0.2, training=self.training)
                x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
                x = F.dropout(x, p=0.2, training=self.training)
                x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
                x = F.dropout(x, p=0.2, training=self.training)
                x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
                x = F.dropout(x, p=0.2, training=self.training)
                x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
                x = F.dropout(x, p=0.2, training=self.training)

                # Official CNN14 global pooling: average over frequency, then
                # combine max and mean over time before the embedding layer.
                x = torch.mean(x, dim=3)
                x1, _ = torch.max(x, dim=2)
                x2 = torch.mean(x, dim=2)
                x = x1 + x2
                x = F.dropout(x, p=0.5, training=self.training)
                embedding = F.relu_(self.fc1(x))  # (B, 2048)
                return embedding

        # ── Load checkpoint ─────────────────────────────────────────────
        model = Cnn14().to(device)
        checkpoint = torch.load(resolved_checkpoint, map_location=device, weights_only=True)

        # The official checkpoint stores weights under the key 'model'
        state_dict = checkpoint.get("model", checkpoint)
        # Never accept a partially loaded acoustic backbone. A previous
        # implementation used different ConvBlock names together with
        # strict=False, leaving all six convolutional blocks random while
        # reporting a successful load.
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self._pann_model = model
        self._pann_device = device
        print(f"[*] PANN CNN14 loaded on {device} from {resolved_checkpoint}")

    def extract_features_pann(
        self,
        audio_clip: np.ndarray,
        checkpoint_path: str | Path | None = None,
    ) -> np.ndarray:
        """
        Runs the audio clip through a pretrained PANN CNN14 backbone and
        returns a 2048-dimensional embedding vector.

        This is a drop-in replacement for extract_features() — the returned
        array is 1-D float32, compatible with TapClassifier.register_tap()
        and TapClassifier.predict() without any other changes.

        Parameters
        ----------
        audio_clip      : np.ndarray
            Raw mono waveform at self.sample_rate. The PANN application path
            captures at 32 kHz; other configured rates are resampled.
        checkpoint_path : str
            Path to the CNN14 .pth checkpoint downloaded from Zenodo.
            Only read on the first call; subsequent calls use the cached model.

        Returns
        -------
        np.ndarray  shape (2048,), dtype float32

        Notes
        -----
        - The model is loaded lazily on the first call unless prepare() was
          called during application startup. It is cached on the instance, so
          the ~2 s PyTorch startup cost is paid once only.
        - Inference runs under torch.no_grad() and the model stays in eval()
          mode throughout, so dropout is disabled and results are deterministic.
        - On Apple Silicon the MPS backend is selected automatically, reducing
          inference time from ~60 ms (CPU) to ~15 ms per clip.
        - PANN was pretrained at 32 kHz. Non-32-kHz callers are resampled.
        """
        import torch

        # Lazy load — only on the very first call
        waveform_array = self._validate_audio(audio_clip)
        if self._pann_model is None:
            selected_checkpoint = (
                self.checkpoint_path if checkpoint_path is None else checkpoint_path
            )
            assert selected_checkpoint is not None
            self._load_pann(str(selected_checkpoint))

        # ── Resample 16 kHz → 32 kHz ────────────────────────────────────
        # PANN's log-mel frontend was computed at 32 kHz during pretraining.
        # Feeding 16 kHz audio without resampling shifts every frequency bin
        # down by one octave, producing a systematically wrong spectrogram.
        PANN_SR = 32000
        if self.sample_rate != PANN_SR:
            audio_32k = librosa.resample(
                waveform_array,
                orig_sr=self.sample_rate,
                target_sr=PANN_SR,
            )
        else:
            audio_32k = waveform_array

        # ── Prepare tensor ───────────────────────────────────────────────
        # CNN14 expects shape (batch, samples). We add the batch dimension,
        # cast to float32 (PyAudio already delivers float32, but librosa
        # resample returns float64 so the cast is always necessary), and move
        # to the same device as the model weights.
        waveform = (
            torch.from_numpy(audio_32k.astype(np.float32))
            .unsqueeze(0)  # (1, samples)
            .to(self._pann_device)
        )

        # ── Forward pass ─────────────────────────────────────────────────
        with torch.no_grad():
            assert self._pann_model is not None
            embedding = self._pann_model(waveform)  # (1, 2048)

        # ── Return as a plain numpy array ────────────────────────────────
        # .cpu() is a no-op when already on CPU, safe to call unconditionally.
        return embedding.squeeze(0).cpu().numpy()  # (2048,)
