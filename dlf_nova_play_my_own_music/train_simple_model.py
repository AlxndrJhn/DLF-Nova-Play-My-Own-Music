import importlib
import os
from datetime import datetime
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from keras.callbacks import ModelCheckpoint  # type: ignore
from keras.utils import to_categorical  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore

from . import simple_model

# change to script dir
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# read datasets
train_dataset = pd.read_pickle("saved_features_train.pickle")
test_dataset = pd.read_pickle("saved_features_test.pickle")

print(f"Got {len(train_dataset)+len(test_dataset)} rows for supervised learning")

# Convert features and corresponding classification labels into numpy arrays
x_train = np.nan_to_num(np.array(train_dataset.feature.tolist()))
y_train = np.array(train_dataset.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
y_train = to_categorical(le.fit_transform(y_train))

x_test = np.nan_to_num(np.array(test_dataset.feature.tolist()))
y_test = np.array(test_dataset.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
y_test = to_categorical(le.fit_transform(y_test))


importlib.reload(simple_model)
model = simple_model.create_model(256)

model.summary()

# Calculate pre-training accuracy
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100 * score[1]

print("Pre-training accuracy: {accuracy:.1f}%")

saved_folder = Path("model_saves")
if not os.path.isdir(saved_folder):
    os.mkdir(saved_folder)

num_epochs = 100
num_batch_size = 256
model_name = "weights.best.simple_cnn_more_features.hdf5"
checkpointer = ModelCheckpoint(
    filepath=str(saved_folder / model_name), verbose=0, save_best_only=True
)
start = datetime.now()

model.fit(
    x_train,
    y_train,
    batch_size=num_batch_size,
    epochs=num_epochs,
    validation_data=(x_test, y_test),
    callbacks=[checkpointer],
    verbose=1,
)

duration = datetime.now() - start
print(f"Training completed in time: {duration}")

# Evaluating the model on the training and testing set
model.load_weights(saved_folder / model_name)
score = model.evaluate(x_train, y_train, verbose=0)
print(f"Training Accuracy: {score[1]*100:.1f}%")

score = model.evaluate(x_test, y_test, verbose=0)
print(f"Testing Accuracy: {score[1]*100:.1f}%")
