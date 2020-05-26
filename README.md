# DLF-Nova Play My Own Music
Switches to Spotify if music is played on DLF Nova, and back when the news start.

# Motivation
The [DLF nova](https://www.deutschlandfunknova.de/) [playlist](https://open.spotify.com/playlist/5qE59dOhR3JtRE8YThsjkO) is just a few hours long, leading to a lot of repetitions if you listen long enough. This is kind of annoying, because the news-bits are quite interesting.

# Requirements
- VLC https://www.videolan.org/vlc/
- spotify account https://www.spotify.com
    - spotify client ID: https://developer.spotify.com/dashboard/applications
- python 3.7

# Get started
1. To use the Spotify API a `client id` and `client secret` are needed, it can be created [here](https://developer.spotify.com/dashboard/applications)
    1. Save `SPOTIPY_CLIENT_ID` in your environment (example `1a2ef9787cc1638261545d6dadb2315`)
    2. Save `SPOTIPY_CLIENT_SECRET` in your environment (example `2c12e5161ace0476c8f2abc70a924ac76`)
2. Find your spotify username and insert it in `spotify_username` in `online_prototype.py`

# Technical details
I trained a simple deep learning model to classify an audio single as "music" or "news", using this classifications, I switch between my spotify device and the online radio stream.

## Dataset
The dataset consists of:
- 500MB of DLF nova podcasts from https://www.deutschlandfunknova.de/podcasts
- [the DLF nova spotify playlist](https://open.spotify.com/playlist/5qE59dOhR3JtRE8YThsjkO) using [spotdl](https://pypi.org/project/spotdl/)
- stream chunks

To split training (80%) and test (20%) datasets, the files are separated randomly before splitting into chunks.
The mp3 files are split into 10s chunks (no overlapping) at 44100Hz.
Each chunk is converted to the feature vector, making the dataset files relatively small.

## Input vector/features
Based on this [this medium article](https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7) and [this article](https://opensource.com/article/19/9/audio-processing-machine-learning-python) I chose three features: `MFCC` (related to speech/music separation, from [librosa](https://pypi.org/project/librosa/)), `GFCC` (typical used for speaker recognition, from [spafe](https://pypi.org/project/spafe/)) and `onset_strength` (related to beat detection in music, from [librosa](https://pypi.org/project/librosa/)). It results in a 256 dimensional vector.

```python
# feature vector
mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
mfccs_scaled = np.mean(mfccs.T, axis=0)

# rather slow
try:
    gfccs = gfcc(audio, num_ceps=20)
    gfccs_scaled = np.mean(gfccs, axis=0)
except Exception:
    gfccs_scaled = np.zeros(20)

hop_length = 2048
oenv = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)

return np.hstack((mfccs_scaled, gfccs_scaled, oenv))
```

Where `audio` is always a 10 second audio chunk at 44100Hz sample rate.

## Deep learning model
I used the [Keras](https://pypi.org/project/Keras/) model from [this medium article](https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7)

```python
model = Sequential()

model.add(Dense(256, input_shape=(input_size,)))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
```

where `input_size` is 256.

It trained for 100 epochs, default learning rate, batch size 256 and saved the weights with the lowest validation loss.

## Performance
It achieved 99.8% accuracy during training and 98% for the validation set.

## Switching
The script `online_prototype.py` registers through the spotify API and requests with OAUTH permission to see the devices and modify the player state. It downloads the DLF stream in 40kB chunks and outputs an classification every 2 to 3 seconds. This signal is smoothed and it triggers switching events. The radio is streamed via VLC.

# Known issues
Some songs are classified as news, due to the simple model
