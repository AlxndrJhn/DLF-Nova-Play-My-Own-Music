import math
from multiprocessing import Pool

import numpy as np  # type: ignore
from pydub import AudioSegment  # type: ignore

from .simple_model_feature import extract_features


class ChopAndProcess(object):
    def __init__(self, chunksize, chunk_offset, audio, sample_rate):
        self.chunksize = chunksize
        self.chunk_offset = chunk_offset
        self.audio = audio
        self.sample_rate = sample_rate

    def __call__(self, i):
        offset = i * self.chunk_offset * 1000
        subaudiosegment = self.audio[offset : offset + self.chunksize * 1000]
        samples = [float(x) for x in subaudiosegment.get_array_of_samples()]
        stacked = np.vstack((samples[0::2], samples[1::2]))
        return extract_features(stacked, self.sample_rate)


def process_single_file(f, chunksize, chunk_offset):
    # load file as audio
    audio = AudioSegment.from_mp3(f)
    sample_rate = 44100
    audio = audio.set_frame_rate(sample_rate)

    last_c_start = len(audio) / 1000 - chunksize
    chunk_n = math.floor(last_c_start / chunk_offset) + 1

    features = []
    with Pool(4) as p:
        results = p.map(
            ChopAndProcess(chunksize, chunk_offset, audio, sample_rate), range(chunk_n)
        )
        features.extend(results)
    return features
