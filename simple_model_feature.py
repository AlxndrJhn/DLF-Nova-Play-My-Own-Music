import librosa
import numpy as np
import soundfile as sf

def extract_features(audio, sample_rate, f=None):
    if f:
        audio, sample_rate = librosa.load(f,mono=False)

    mfccs1 = librosa.feature.mfcc(y=np.asfortranarray(audio[0,:]), sr=sample_rate, n_mfcc=40)
    mfccsscaled1 = np.mean(mfccs1.T,axis=0)

    # mfccs2 = librosa.feature.mfcc(y=np.asfortranarray(audio[1,:]), sr=sample_rate, n_mfcc=40)
    # mfccsscaled2 = np.mean(mfccs2.T,axis=0)

    return mfccsscaled1 #np.hstack((mfccsscaled1, mfccsscaled2))
