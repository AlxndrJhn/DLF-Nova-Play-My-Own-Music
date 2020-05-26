# %%
import pandas
from pathlib import Path
import numpy as np
from pydub import AudioSegment

import simple_model_feature

url = "https://dradio-edge-209a-fra-lg-cdn.cast.addradio.de/dradio/nova/live/mp3/128/stream.mp3"  # nova
# url = "http://st01.dlf.de/dlf/01/128/mp3/stream.mp3" # DLF
# url = "https://st02.sslstream.dlf.de/dlf/02/128/mp3/stream.mp3" # kultur

dataset_path = Path("nova_classifier") / "datasets"

dataset_id = 1
chunk_length = 10  # seconds
chunk_move_step = 10  # seconds

path_labels = dataset_path / f"{dataset_id:02}.txt"
path_mp3 = dataset_path / f"{dataset_id:02}.mp3"
chunk_path = dataset_path / f"{dataset_id:02}_split"

# %%
# load model
from complex2_model import load_trained_model

model = load_trained_model("nova_classifier/model_saves/weights.best.deeper2_cnn_40.hdf5")
model.summary()


# %%
# test with validation dataset
import pandas as pd

featuresdf = pd.read_pickle("nova_classifier/saved_features_test.pickle")

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
x_test = np.array(featuresdf.feature.tolist())
y_test = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
y_test = to_categorical(le.fit_transform(y_test))

# split the dataset
from sklearn.model_selection import train_test_split

score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100 * score[1]

print("Model accuracy: %.4f%%" % accuracy)

# %%
# online
def classify_sound_chunk(sound_chunk):
    sound_chunk = sound_chunk.set_frame_rate(44100)
    channels = 2
    samples = [float(x) for x in sound_chunk.get_array_of_samples()]
    stacked = np.vstack((samples[0::channels], samples[1::channels]))
    data = simple_model_feature.extract_features(stacked, sound_chunk.frame_rate)

    p = model.predict_proba(data.reshape(1, 40))[0]

    if p[0] > p[1]:
        prediction_label = "m"
        probability = p[0]
    else:
        prediction_label = "p"
        probability = p[1]
    return prediction_label, probability


# %%
import spotipy
import spotipy.util as util

scope = "user-modify-playback-state"
username = "jahn.alexander"
token = ''
sp = ''
def refresh_spotify():
    global token
    global sp
    token = util.prompt_for_user_token(username, scope, redirect_uri="http://127.0.0.1:9090")
    sp = spotipy.Spotify(auth=token)
refresh_spotify()

#%%
def stop_music():
    try:
        sp.pause_playback()
    except:
        pass


def start_music():
    try:
        sp.start_playback()
    except:
        refresh_spotify()
        sp.start_playback()

stop_music()

# %%
import vlc

Instance = vlc.Instance()
player = Instance.media_player_new()


def start_radio():
    Media = Instance.media_new(url)
    Media.get_mrl()
    player.set_media(Media)
    player.play()


def stop_radio():
    player.stop()


# %%
from io import BytesIO
from urllib.request import urlopen
from IPython.display import display, clear_output
from datetime import datetime

u = urlopen(url, timeout=5)
buffer = []
radio_states = []
radio_state_n = 8
play_music = False
switch_signal = 0
switch_signal_factor = 0.6
first_prediction = True
while True:
    try:
        data = u.read(1024 * 40)
        if data == b"":
            u = urlopen(url, timeout=5)
            continue
        audio_segment = AudioSegment.from_mp3(BytesIO(data))
        buffer.append(audio_segment)
    except Exception as e:
        print(f"{datetime.now()} error: {e}")
        u = urlopen(url, timeout=5)
        continue

    concat_audio = sum(buffer)

    missing = chunk_length - len(concat_audio) / 1000
    if missing > 0:
        print(f"{datetime.now()} {missing: 4.1f}s missing")
        continue

    # got material
    cropped = concat_audio[-10000:]
    buffer = [cropped]
    start = datetime.now()
    classification, probability = classify_sound_chunk(cropped)
    performance = 1 / (datetime.now() - start).microseconds * 1000000

    class_val = 0 if classification == "m" else 1
    switch_signal = switch_signal_factor * switch_signal + (1 - switch_signal_factor) * class_val

    mapping = {"p": "news", "m": "music"}

    switch_msg = ""
    date = str(datetime.now()).replace(':','-').replace('.', '-')
    file_name = f"{date}_{mapping[classification]}.mp3"
    if (play_music and switch_signal > 0.9) or (first_prediction and class_val == 1):
        first_prediction = False
        play_music = False
        switch_msg = "switch to radio"
        stop_music()
        start_radio()
        cropped.export(file_name, format="mp3")

    elif (not play_music and switch_signal < 0.1) or (first_prediction and class_val == 0):
        first_prediction = False
        play_music = True
        switch_msg = "switch to music"
        stop_radio()
        start_music()
        cropped.export(file_name, format="mp3")

    print(
        f"{datetime.now()} Classification: {mapping[classification]:>6}, prob.: {probability*100:3.0f}%, filtered signal: {switch_signal*100: 3.0f}%, AI performance: {performance:4.1f}Hz {switch_msg}"
    )

# %%
