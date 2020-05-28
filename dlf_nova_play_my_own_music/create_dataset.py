import importlib
import random
from pathlib import Path

import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore

from . import process_mp3_file

train_ratio = 80  # %
chunksize = 10  # s
chunk_offset = 10  # s
dataset_folder = Path("nova_classifier") / "datasets"
num_processors = 7
random_seed = 42


if __name__ == "__main__":
    importlib.reload(process_mp3_file)

    features_train = []
    features_test = []
    for label in ["m", "p"]:
        files = list((dataset_folder / label).iterdir())
        random.Random(random_seed).shuffle(files)

        # train data
        n_train = round(len(files) * train_ratio / 100)
        n_test = len(files) - n_train

        # train data
        for f in tqdm(files[:n_train]):
            output = process_mp3_file.process_single_file(
                f, chunksize=chunksize, chunk_offset=chunk_offset
            )
            for chunk_feature in output:
                features_train.append([chunk_feature, label])

        # test data
        for f in tqdm(files[n_train : n_train + n_test]):
            output = process_mp3_file.process_single_file(
                f, chunksize=chunksize, chunk_offset=chunk_offset
            )
            for chunk_feature in output:
                features_test.append([chunk_feature, label])

        features_train_df = pd.DataFrame(
            features_train, columns=["feature", "class_label"]
        )
        features_train_df.to_pickle("nova_classifier/saved_features_train.pickle")
        print(f"Saved {len(features_train_df)} rows to disk for training")

        features_test_df = pd.DataFrame(
            features_test, columns=["feature", "class_label"]
        )
        features_test_df.to_pickle("nova_classifier/saved_features_test.pickle")
        print(f"Saved {len(features_train_df)} rows to disk for testing")
