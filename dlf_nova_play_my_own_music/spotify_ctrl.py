from time import sleep
from typing import Callable

import spotipy.util as util  # type: ignore
from spotipy import Spotify


class SpotifyManager:
    sci: str  # spotify client id
    scs: str  # spotify client secret
    su: str  # spotify username
    sp: Spotify
    device_id: str

    def __init__(
        self,
        spotify_client_id: str,
        spotify_client_secret: str,
        spotify_username: str,
        log: Callable[[str], None],
    ):
        self.sci = spotify_client_id
        self.scs = spotify_client_secret
        self.su = spotify_username
        self.retries = 3
        self.log = log

    def connect(self):
        # get token
        scope = "user-read-playback-state,user-modify-playback-state"
        token = util.prompt_for_user_token(
            username=self.su,
            scope=scope,
            client_id=self.sci,
            client_secret=self.scs,
            redirect_uri="http://127.0.0.1:9090",
        )
        self.sp = Spotify(auth=token)

        # get device id
        res = self.sp.devices()
        if len(res) == 0:
            raise Exception("No spotify device found, play something somewhere")
        self.device_id = res["devices"][0]["id"]

    def stop_music(self):
        for attempt in range(self.retries):
            try:
                self.sp.pause_playback(device_id=self.device_id)
            except Exception as e:
                self.log(f"Error starting music: {e}, retrying")
                self.connect()
            else:
                break
        else:
            self.log("Error starting music, sleeping for 1 minute")
            sleep(60)

    def start_music(self):
        for attempt in range(self.retries):
            try:
                self.sp.start_playback(device_id=self.device_id)
            except Exception as e:
                self.log(f"Error starting music: {e}, retrying")
                self.connect()
            else:
                break
        else:
            self.log("Error starting music, sleeping for 1 minute")
            sleep(60)
