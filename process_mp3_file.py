# %%
from simple_model_feature import extract_features
from pydub import AudioSegment
import numpy as np
import math

# %%
def process_single_file(f, chunksize=10, chunk_offset=4):
    # load file as audio
    audio = AudioSegment.from_mp3(f)
    sample_rate = 44100
    audio = audio.set_frame_rate(sample_rate)
    samples = [float(x) for x in audio.get_array_of_samples()]
    stacked = np.vstack((samples[0::2], samples[1::2]))

    last_c_start = len(audio) / 1000 - chunksize
    chunk_n = math.floor(last_c_start / chunk_offset) + 1

    length = chunksize * sample_rate
    features = []
    for i in range(chunk_n):
        offset = i * chunk_offset * sample_rate
        subsample = stacked[:, offset : offset + length]
        features.append(extract_features(subsample, sample_rate))
    return features


# %%
#features = process_single_file(r"C:\Users\jahna\sandbox_37\nova_classifier\datasets\m\COSBY - Spaceship.mp3", 10, 10)
