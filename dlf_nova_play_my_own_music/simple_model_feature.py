from time import time
from typing import Optional

import librosa  # type: ignore
import numpy as np  # type: ignore
from spafe.features.gfcc import gfcc  # type: ignore


def extract_features(
    audio: np.ndarray, sample_rate: int, f: Optional[str] = None
) -> np.ndarray:
    if f:
        audio, sample_rate = librosa.load(f, mono=False)

    # stereo to mono
    if len(audio.shape) == 2:
        audio = audio.sum(axis=0) / 2

    audio = np.asfortranarray(audio)

    # feature vector
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    mfccs_scaled = np.mean(mfccs.T, axis=0)

    # rather slow
    try:
        gfccs = gfcc(audio, num_ceps=20)
        gfccs_scaled = np.mean(gfccs, axis=0)
    except Exception:
        gfccs_scaled = np.zeros(20)

    hop_length = 2048
    oenv = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)

    return np.hstack((mfccs_scaled, gfccs_scaled, oenv))


if __name__ == "__main__":
    y, sr = librosa.load(librosa.util.example_audio_file(), duration=10)
    start = time()
    res = extract_features(y, sr)
    print(f"Performance: {1/(time()-start):.1f}Hz")
