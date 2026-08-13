import numpy as np
import librosa

class SignalProcessor:
    """
    Handles transient detection (counting taps) and feature extraction
    (generating a sonic fingerprint) for incoming audio buffers.
    """
    def __init__(self, sample_rate: int = 16000, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def calculate_energy(self, audio_frame: np.ndarray) -> float:
        """
        Calculates the Root Mean Square (RMS) energy of a raw audio frame.
        Used by the listener thread to detect if a sound broke the noise floor.
        """
        if len(audio_frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_frame**2)))

    def count_taps(self, audio_clip: np.ndarray, debounce_seconds: float = 0.05) -> int:
        """
        Detects sudden spikes in acoustic energy to count discrete impacts.
        Includes a debounce parameter to prevent room echo or muddy strikes 
        from being counted as multiple taps.
        """
        _, percussive = librosa.effects.hpss(audio_clip)
        # Convert the debounce time (seconds) into the number of frames required by librosa
        wait_frames = int((debounce_seconds * self.sample_rate) / self.hop_length)

        onsets = librosa.onset.onset_detect(
            y=percussive,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            wait=wait_frames,
            backtrack=False
        )
        # print(onsets)
        return len(onsets)
    
    def extract_features_yamnet(self, audio_clip: np.ndarray):  
        return None
    
    def extract_features_pann(self, audio_clip: np.ndarray):  
        return None

    def extract_features(self, audio_clip: np.ndarray) -> np.ndarray:
        """
        Compresses a 1.5-second audio array into a flat, 1D feature vector
        representing the acoustic profile of the tap.
        """

        harmonic, percussive = librosa.effects.hpss(audio_clip)
        # 1. Mel-Frequency Cepstral Coefficients (Timbre/Material of the tap)
        # 13 coefficients is standard for non-speech audio classification
        mfccs = librosa.feature.mfcc(y=percussive, sr=self.sample_rate, n_mfcc=13)
        
        # 2. Spectral Centroid (Brightness/Sharpness of the tap)
        centroid = librosa.feature.spectral_centroid(y=percussive, sr=self.sample_rate)
        
        # 3. Spectral Rolloff (High-frequency decay)
        rolloff = librosa.feature.spectral_rolloff(y=percussive, sr=self.sample_rate)
        
        # Compress the 2D time-series arrays into flat 1D statistical summaries
        # We take the mean and standard deviation to capture the 'shape' of the sound over time
        feature_vector = np.hstack((
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(centroid), np.std(centroid),
            np.mean(rolloff), np.std(rolloff)
        ))
        
        return feature_vector