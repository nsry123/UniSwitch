import numpy as np
import librosa

class SignalProcessor:
    """
    Handles transient detection (counting taps) and feature extraction
    (generating a sonic fingerprint) for incoming audio buffers.
    """
    def __init__(self, model = "Default",sample_rate: int = 16000, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length  = hop_length
        self.model = model
        if model == "pann":
            self.sample_rate = 32000

        # PANN model and its transforms are loaded lazily on first call to
        # extract_features_pann(). This avoids a multi-second PyTorch startup
        # cost when the class is instantiated in contexts that don't use PANN.
        self._pann_model      = None
        self._pann_to_mel     = None
        self._pann_device     = None

    # ------------------------------------------------------------------ #
    #  Energy gate (used by the listener thread)                          #
    # ------------------------------------------------------------------ #

    def calculate_energy(self, audio_frame: np.ndarray) -> float:
        """
        Calculates the Root Mean Square (RMS) energy of a raw audio frame.
        Used by the listener thread to detect if a sound broke the noise floor.
        """
        if len(audio_frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_frame**2)))

    # ------------------------------------------------------------------ #
    #  Knock counting (onset detection — unchanged)                       #
    # ------------------------------------------------------------------ #

    def count_taps(self, audio_clip: np.ndarray, debounce_seconds: float = 0.05) -> int:
        """
        Detects sudden spikes in acoustic energy to count discrete impacts.
        Includes a debounce parameter to prevent room echo or muddy strikes
        from being counted as multiple taps.
        """
        _, percussive = librosa.effects.hpss(audio_clip)
        wait_frames   = int((debounce_seconds * self.sample_rate) / self.hop_length)

        onsets = librosa.onset.onset_detect(
            y=percussive,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            wait=wait_frames,
            backtrack=False,
        )
        print(onsets)
        return len(onsets)

    # ------------------------------------------------------------------ #
    #  Legacy MFCC feature extractor (kept for backwards compatibility)   #
    # ------------------------------------------------------------------ #

    def extract_features(self, audio_clip: np.ndarray) -> np.ndarray:
        if self.model == "pann":
            return self.extract_features_pann(audio_clip=audio_clip)
        else:
            return self.extract_features_default(audio_clip=audio_clip)


    def extract_features_default(self, audio_clip: np.ndarray) -> np.ndarray:
        """
        Compresses a 1.5-second audio array into a flat, 1D feature vector
        representing the acoustic profile of the tap.
        """
        harmonic, percussive = librosa.effects.hpss(audio_clip)

        mfccs    = librosa.feature.mfcc(y=percussive,  sr=self.sample_rate, n_mfcc=13)
        centroid = librosa.feature.spectral_centroid(y=audio_clip, sr=self.sample_rate)
        rolloff  = librosa.feature.spectral_rolloff(y=audio_clip,  sr=self.sample_rate)

        feature_vector = np.hstack((
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(centroid),      np.std(centroid),
            np.mean(rolloff),       np.std(rolloff),
        ))
        return feature_vector

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
        from torchlibrosa.stft       import Spectrogram, LogmelFilterBank
        from torchlibrosa.augmentation import SpecAugmentation

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
        SAMPLE_RATE = 32000   # PANN was pretrained at 32 kHz
        WINDOW_SIZE = 1024
        HOP_SIZE    = 320
        MEL_BINS    = 64
        FMIN        = 50
        FMAX        = 14000

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
                self.bn1   = nn.BatchNorm2d(out_ch)
                self.bn2   = nn.BatchNorm2d(out_ch)
                init_layer(self.conv1); init_layer(self.conv2)
                init_bn(self.bn1);      init_bn(self.bn2)

            def forward(self, x, pool_size=(2, 2), pool_type="avg"):
                x = F.relu_(self.bn1(self.conv1(x)))
                x = F.relu_(self.bn2(self.conv2(x)))
                if   pool_type == "max":  x = F.max_pool2d(x, pool_size)
                elif pool_type == "avg":  x = F.avg_pool2d(x, pool_size)
                elif pool_type == "avg+max":
                    x = F.avg_pool2d(x, pool_size) + F.max_pool2d(x, pool_size)
                return x

        class Cnn14(nn.Module):
            def __init__(self):
                super().__init__()
                self.spectrogram_extractor = spectrogram_extractor
                self.logmel_extractor      = logmel_extractor
                self.bn0         = nn.BatchNorm2d(64)
                self.conv_block1 = ConvBlock(1,   64)
                self.conv_block2 = ConvBlock(64,  128)
                self.conv_block3 = ConvBlock(128, 256)
                self.conv_block4 = ConvBlock(256, 512)
                self.conv_block5 = ConvBlock(512, 1024)
                self.conv_block6 = ConvBlock(1024, 2048)
                self.fc1         = nn.Linear(2048, 2048, bias=True)
                self.fc_audioset = nn.Linear(2048, 527, bias=True)
                init_bn(self.bn0)
                init_layer(self.fc1)
                init_layer(self.fc_audioset)

            def forward(self, waveform: "torch.Tensor"):
                x = self.spectrogram_extractor(waveform)   # (B, 1, T, F)
                x = self.logmel_extractor(x)                # (B, 1, T, 64)

                x = x.transpose(1, 3)                       # (B, 64, T, 1)
                x = self.bn0(x)
                x = x.transpose(1, 3)                       # (B, 1, T, 64)

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
                x  = x1 + x2
                x  = F.dropout(x, p=0.5, training=self.training)
                embedding = F.relu_(self.fc1(x))             # (B, 2048)
                return embedding

        # ── Load checkpoint ─────────────────────────────────────────────
        model = Cnn14().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # The official checkpoint stores weights under the key 'model'
        state_dict = checkpoint.get("model", checkpoint)
        # Never accept a partially loaded acoustic backbone. A previous
        # implementation used different ConvBlock names together with
        # strict=False, leaving all six convolutional blocks random while
        # reporting a successful load.
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self._pann_model  = model
        self._pann_device = device
        print(f"[*] PANN CNN14 loaded on {device} from {checkpoint_path}")

    def extract_features_pann(
        self,
        audio_clip: np.ndarray,
        checkpoint_path: str = "models/Cnn14_mAP=0.431.pth",
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
            Raw float32 waveform at self.sample_rate (16 kHz from the listener).
            The clip is automatically resampled to 32 kHz to match the PANN
            pretraining sample rate before the forward pass.
        checkpoint_path : str
            Path to the CNN14 .pth checkpoint downloaded from Zenodo.
            Only read on the first call; subsequent calls use the cached model.

        Returns
        -------
        np.ndarray  shape (2048,), dtype float32

        Notes
        -----
        - The model is loaded lazily on the first call and cached on the
          instance, so the ~2 s PyTorch startup cost is paid once only.
        - Inference runs under torch.no_grad() and the model stays in eval()
          mode throughout, so dropout is disabled and results are deterministic.
        - On Apple Silicon the MPS backend is selected automatically, reducing
          inference time from ~60 ms (CPU) to ~15 ms per clip.
        - PANN was pretrained at 32 kHz. The listener captures at 16 kHz.
          We resample here rather than changing the listener, keeping the rest
          of the pipeline (onset detection, energy gate) at 16 kHz where
          librosa's defaults are well-tuned.
        """
        import torch

        # Lazy load — only on the very first call
        if self._pann_model is None:
            self._load_pann(checkpoint_path)

        # ── Resample 16 kHz → 32 kHz ────────────────────────────────────
        # PANN's log-mel frontend was computed at 32 kHz during pretraining.
        # Feeding 16 kHz audio without resampling shifts every frequency bin
        # down by one octave, producing a systematically wrong spectrogram.
        PANN_SR = 32000
        if self.sample_rate != PANN_SR:
            audio_32k = librosa.resample(
                audio_clip,
                orig_sr=self.sample_rate,
                target_sr=PANN_SR,
            )
        else:
            audio_32k = audio_clip

        # ── Prepare tensor ───────────────────────────────────────────────
        # CNN14 expects shape (batch, samples). We add the batch dimension,
        # cast to float32 (PyAudio already delivers float32, but librosa
        # resample returns float64 so the cast is always necessary), and move
        # to the same device as the model weights.
        waveform = (
            torch.from_numpy(audio_32k.astype(np.float32))
            .unsqueeze(0)                        # (1, samples)
            .to(self._pann_device)
        )

        # ── Forward pass ─────────────────────────────────────────────────
        with torch.no_grad():
            embedding = self._pann_model(waveform)   # (1, 2048)

        # ── Return as a plain numpy array ────────────────────────────────
        # .cpu() is a no-op when already on CPU, safe to call unconditionally.
        return embedding.squeeze(0).cpu().numpy()    # (2048,)
