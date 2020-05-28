
from typing import Tuple

import numpy as np  # type: ignore
from keras.models import Model  # type: ignore
from pydub import AudioSegment  # type: ignore

from . import simple_model_feature
from .constants import INPUT_LAYER_SIZE


def classify_sound_chunk(model: Model, sound_chunk: AudioSegment) -> Tuple[str, float]:
    """Preprocess sound chunk and use the model to predict a label."""
    # preprocess
    channels = 2
    sound_chunk = sound_chunk.set_frame_rate(44100)
    samples = [float(x) for x in sound_chunk.get_array_of_samples()]
    stacked = np.vstack((samples[0::channels], samples[1::channels]))
    f_vector = simple_model_feature.extract_features(stacked, sound_chunk.frame_rate)

    # predict
    p = model.predict_proba(f_vector.reshape(1, INPUT_LAYER_SIZE))[0]

    # transform to label
    if p[0] > p[1]:
        prediction_label = "m"
        probability = p[0]
    else:
        prediction_label = "p"
        probability = p[1]
    return prediction_label, probability
