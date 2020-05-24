# %%
import importlib
import random
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# %%
import process_mp3_file

# %%
train_ratio = 80  # %
chunksize = 10  # s
chunk_offset = 1  # s
dataset_folder = Path("nova_classifier") / "datasets"
num_processors = 8
random_seed = 42

# %%
if __name__ == "__main__":
    importlib.reload(process_mp3_file)

    with Pool(num_processors) as p:
        features_train = []
        features_test = []
        for label in ["m", "p"]:
            files = list((dataset_folder / label).iterdir())
            random.Random(random_seed).shuffle(files)

            # train data
            n_train = round(len(files) * train_ratio / 100)
            n_test = len(files) - n_train
            output = list(tqdm(p.imap(process_mp3_file.process_single_file, files[:n_train]), total=n_train))
            for o in output:
                for chunk_feature in o:
                    features_train.append([chunk_feature, label])

            # test data
            output = list(tqdm(p.imap(process_mp3_file.process_single_file, files[n_train : n_train + n_test]), total=n_test))
            for o in output:
                for chunk_feature in o:
                    features_test.append([chunk_feature, label])

        features_train_df = pd.DataFrame(features_train, columns=["feature", "class_label"])
        features_train_df.to_pickle("saved_features_train.pickle")
        print(f"Saved {len(features_train_df)} rows to disk for training")

        features_test_df = pd.DataFrame(features_test, columns=["feature", "class_label"])
        features_test_df.to_pickle("saved_features_test.pickle")
        print(f"Saved {len(features_train_df)} rows to disk for testing")
# %%
# output[0].shape
