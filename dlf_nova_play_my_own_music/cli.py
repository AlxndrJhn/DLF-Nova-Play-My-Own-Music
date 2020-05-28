import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from time import sleep
from urllib.request import urlopen

import click
import vlc  # type: ignore
from pydub import AudioSegment  # type: ignore

from .constants import (
    CHUNK_LENGTH,
    INPUT_LAYER_SIZE,
    MODEL_FOLDER,
    MODEL_NAME,
    SWITCH_SIGNAL_FACTOR,
)
from .spotify_ctrl import SpotifyManager

# force Keras to use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@click.command()
@click.option(
    "-sci",
    "--spot_client_id",
    envvar="SPOTIPY_CLIENT_ID",
    help="Spotify client id (checks env. var `SPOTIPY_CLIENT_ID` as well)",
)
@click.option(
    "-scs",
    "--spot_client_secret",
    envvar="SPOTIPY_CLIENT_SECRET",
    help="Spotify client secret (checks env. var `SPOTIPY_CLIENT_SECRET` as well)",
)
@click.option(
    "-su",
    "--spot_username",
    envvar="SPOTIPY_USERNAME",
    help="Spotify username (checks env. var `SPOTIPY_USERNAME` as well)",
)
@click.option(
    "-s",
    "--stream",
    default="https://dradio-edge-209a-fra-lg-cdn.cast.addradio.de/dradio/nova/live/mp3/128/stream.mp3",  # noqa
    help="Stream url, default is nova",
)
@click.option(
    "-cof",
    "--chunk-output-folder",
    help="Set to a path to save all transition audio chunks for manual review.",
)
@click.option("-v", "--verbose", count=True)
def main(
    spot_client_id,
    spot_client_secret,
    spot_username,
    stream,
    chunk_output_folder,
    verbose,
):
    """Starts the automatic music/news detection and starts streaming"""

    # connect spotify
    spm: SpotifyManager = SpotifyManager(
        spot_client_id, spot_client_secret, spot_username, log
    )
    spm.connect()

    # VLC control for radio stream
    vlc_instance = vlc.Instance()
    player = vlc_instance.media_player_new()

    def start_radio():
        Media = vlc_instance.media_new(stream)
        Media.get_mrl()
        player.set_media(Media)
        player.play()

    def stop_radio():
        player.stop()

    # new chunk folder option
    if chunk_output_folder:
        new_audio_chunk_folder = Path(chunk_output_folder)
        new_audio_chunk_folder.mkdir(exist_ok=True)

    # load model, late import because tf is slow af
    from .simple_model import load_trained_model
    from .prediction import classify_sound_chunk

    abspath = os.path.abspath(__file__)
    dir_name = os.path.dirname(abspath)
    model = load_trained_model(
        Path(dir_name) / MODEL_FOLDER / MODEL_NAME, INPUT_LAYER_SIZE
    )
    if verbose >= 1:
        model.summary()

    # main loop
    u = urlopen(stream, timeout=5)
    buffer = []
    is_playing_music = False
    switch_signal = 0
    first_prediction = True
    failure_counter = 0
    while True:
        # fetch 40k of data from the stream
        try:
            data = u.read(1024 * 40)

            # should never be empty
            if data == b"":
                failure_counter += 1

                if failure_counter >= 3:
                    log("Having issues with stream, sleeping 10s")
                    sleep(10)

                u = urlopen(stream, timeout=5)
                continue

            # convert to audio and append to buffer
            audio_segment = AudioSegment.from_mp3(BytesIO(data))
            buffer.append(audio_segment)
        except Exception as e:
            log(f"error: {e}")
            u = urlopen(stream, timeout=5)
            continue

        # concat the audio elements to longer audio
        concat_audio = sum(buffer)

        # expect at least 10s of audio
        missing = CHUNK_LENGTH - len(concat_audio) / 1000
        if missing > 0:
            log(f"{missing: 4.1f}s missing")
            continue

        # got enough material, take the last 10000ms of it
        cropped = concat_audio[-10000:]

        # throw away the older audio
        buffer = [cropped]

        # classify the audio
        start = datetime.now()
        classification, probability = classify_sound_chunk(model, cropped)
        performance = 1 / (datetime.now() - start).microseconds * 1000000

        # map class to a state input
        class_val = 0 if classification == "m" else 1

        # exponential filtering
        switch_signal = (
            SWITCH_SIGNAL_FACTOR * switch_signal
            + (1 - SWITCH_SIGNAL_FACTOR) * class_val
        )

        mapping = {"p": "news", "m": "music"}

        # to save the transition snippets for new dataset
        date = str(datetime.now()).replace(":", "-").replace(".", "-")
        file_name = f"{date}_{mapping[classification]}.mp3"

        # check the thresholds to switch the stream/spotify
        switch_msg = ""
        if (is_playing_music and switch_signal > 0.9) or (
            first_prediction and class_val == 1
        ):
            first_prediction = False
            is_playing_music = False
            switch_msg = "switch to radio"
            spm.stop_music()
            start_radio()

            if chunk_output_folder:
                cropped.export(new_audio_chunk_folder / file_name, format="mp3")

        elif (not is_playing_music and switch_signal < 0.1) or (
            first_prediction and class_val == 0
        ):
            first_prediction = False
            is_playing_music = True
            switch_msg = "switch to music"
            stop_radio()
            spm.start_music()

            if chunk_output_folder:
                cropped.export(new_audio_chunk_folder / file_name, format="mp3")

        class_txt = f"Class: {mapping[classification]:>6}, "
        proba_txt = f"prob.: {probability*100:3.0f}%, "
        state = f"filtered signal: {switch_signal*100: 3.0f}%, "
        perf = f"AI performance: {performance:4.1f}Hz {switch_msg}"
        log(class_txt + proba_txt + state + perf)


def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    click.echo(f"{now}: {msg}")


if __name__ == "__main__":
    main()
