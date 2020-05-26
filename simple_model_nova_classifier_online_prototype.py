
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
# test with validation dataset
import pandas as pd
import spotipy
import spotipy.util as util
import vlc
from IPython.display import clear_output, display
from keras.utils import to_categorical
from pydub import AudioSegment
# split the dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import simple_model_feature
from simple_model import load_trained_model

# parameters
url = "https://dradio-edge-209a-fra-lg-cdn.cast.addradio.de/dradio/nova/live/mp3/128/stream.mp3"  # nova
# url = "http://st01.dlf.de/dlf/01/128/mp3/stream.mp3" # DLF
# url = "https://st02.sslstream.dlf.de/dlf/02/128/mp3/stream.mp3" # kultur

chunk_length = 10  # seconds
input_layer_size = 256
model_name = 'weights.best.simple_cnn_more_features.hdf5'
spotify_username = "jahn.alexander"
switch_signal_factor = 0.3 # 0.1 quick switch, 0.9 slow switch

# load model
model = load_trained_model(f"nova_classifier/model_saves/{model_name}", input_layer_size)
model.summary()

# confirm performance with test dataset
featuresdf = pd.read_pickle("nova_classifier/saved_features_test.pickle")

# Convert features and corresponding classification labels into numpy arrays
x_test = np.array(featuresdf.feature.tolist())
y_test = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
y_test = to_categorical(le.fit_transform(y_test))

# scoring
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100 * score[1]

print(f"Model accuracy: {accuracy:.1f}%")

def classify_sound_chunk(sound_chunk):
    sound_chunk = sound_chunk.set_frame_rate(44100)
    channels = 2
    samples = [float(x) for x in sound_chunk.get_array_of_samples()]
    stacked = np.vstack((samples[0::channels], samples[1::channels]))
    data = simple_model_feature.extract_features(stacked, sound_chunk.frame_rate)

    p = model.predict_proba(data.reshape(1, 256))[0]

    if p[0] > p[1]:
        prediction_label = "m"
        probability = p[0]
    else:
        prediction_label = "p"
        probability = p[1]
    return prediction_label, probability

# spotify interface, need to have a device playing to take control
scope = "user-modify-playback-state"
token = ''
sp = ''
def refresh_spotify():
    global token
    global sp
    token = util.prompt_for_user_token(spotify_username, scope, redirect_uri="http://127.0.0.1:9090")
    sp = spotipy.Spotify(auth=token)
refresh_spotify()

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

# VLC control for radio stream
Instance = vlc.Instance()
player = Instance.media_player_new()

def start_radio():
    Media = Instance.media_new(url)
    Media.get_mrl()
    player.set_media(Media)
    player.play()

def stop_radio():
    player.stop()

# fetch chunks of the audio-stream
u = urlopen(url, timeout=5)
buffer = []
radio_states = []
play_music = False
switch_signal = 0
first_prediction = True
while True:
    # fetch 40k of data from the stream
    try:
        data = u.read(1024 * 40)

        # should never be empty
        if data == b"":
            u = urlopen(url, timeout=5)
            continue

        # convert to audio and append to buffer
        audio_segment = AudioSegment.from_mp3(BytesIO(data))
        buffer.append(audio_segment)
    except Exception as e:
        print(f"{datetime.now()} error: {e}")
        u = urlopen(url, timeout=5)
        continue

    # concat the audio elements to longer audio
    concat_audio = sum(buffer)

    # expect at least 10s of audio
    missing = chunk_length - len(concat_audio) / 1000
    if missing > 0:
        print(f"{datetime.now()} {missing: 4.1f}s missing")
        continue

    # got enough material, take the last 10000ms of it
    cropped = concat_audio[-10000:]

    # throw away the older audio
    buffer = [cropped]

    # classify the audio
    start = datetime.now()
    classification, probability = classify_sound_chunk(cropped)
    performance = 1 / (datetime.now() - start).microseconds * 1000000

    # map class to a state input
    class_val = 0 if classification == "m" else 1

    # exponential filtering
    switch_signal = switch_signal_factor * switch_signal + (1 - switch_signal_factor) * class_val

    mapping = {"p": "news", "m": "music"}

    # to save the transition snippets for new dataset
    date = str(datetime.now()).replace(':','-').replace('.', '-')
    file_name = f"{date}_{mapping[classification]}.mp3"

    # check the thresholds to switch the stream/spotify
    switch_msg = ""
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
