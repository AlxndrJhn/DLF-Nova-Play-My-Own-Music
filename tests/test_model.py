from pathlib import Path
from time import time

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore

from dlf_nova_play_my_own_music.constants import INPUT_LAYER_SIZE
from dlf_nova_play_my_own_music.process_mp3_file import process_single_file


def test_accuracy_validation_set():
    from dlf_nova_play_my_own_music.simple_model import load_trained_model
    from keras.utils import to_categorical  # type: ignore

    base_path = Path(__file__).resolve().parents[1] / "dlf_nova_play_my_own_music"
    # get dataset
    dataset_path = base_path / "saved_features_test.pickle"
    test_dataset = pd.read_pickle(dataset_path)
    assert len(test_dataset) > 200

    x_test = np.nan_to_num(np.array(test_dataset.feature.tolist()))
    y_test = np.array(test_dataset.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    y_test = to_categorical(le.fit_transform(y_test))
    model = load_trained_model(
        base_path / "model_saves" / "weights.best.simple_cnn_more_features.hdf5",
        INPUT_LAYER_SIZE,
    )
    start_time = time()
    score = model.evaluate(x_test, y_test, verbose=1)
    duration = (time() - start_time) / len(test_dataset)
    accuracy = score[1]
    assert accuracy > 0.98
    assert duration < 0.001


def test_feature_performace(shared_datadir):
    f = shared_datadir / "2020-05-26 15-47-05-113724_news.mp3"
    start_time = time()
    feature = process_single_file(f, 10, 10)
    duration = time() - start_time
    assert len(feature) == 1
    assert len(feature[0]) == INPUT_LAYER_SIZE
    assert sum(feature[0]) != 0
    assert 0.5 < duration < 4


def test_create_model_no_weights():
    from dlf_nova_play_my_own_music.simple_model import create_model

    model = create_model(INPUT_LAYER_SIZE)
    model.layers
    assert model is not None
