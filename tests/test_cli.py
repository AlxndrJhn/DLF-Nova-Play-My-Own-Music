from io import BytesIO

import pytest  # type: ignore
from click.testing import CliRunner
from pydub import AudioSegment  # type: ignore

from dlf_nova_play_my_own_music import __version__, cli


def test_version():
    assert __version__ == "0.1.3"


@pytest.fixture
def mock_bar(mocker):
    return mocker.patch("cli.SpotifyManager")


def test_typical_application(monkeypatch, shared_datadir):
    # mock spotify
    spot_tried_to_connect = False
    spot_tried_to_start = False
    spot_tried_to_stop = False

    class mocked_spot:
        def connect(self):
            nonlocal spot_tried_to_connect
            spot_tried_to_connect = True

        def start_music(self):
            nonlocal spot_tried_to_start
            spot_tried_to_start = True

        def stop_music(self):
            nonlocal spot_tried_to_stop
            spot_tried_to_stop = True

    def mock_spot(*args, **kwargs):
        return mocked_spot()

    monkeypatch.setattr(cli, "SpotifyManager", mock_spot)

    # mock vlc
    vlc_tried_to_connect = False
    vlc_tried_to_start = False
    vlc_tried_to_stop = False
    vlc_media_url = None

    class mocked_vlc_media:
        def get_mrl(self):
            pass

    class mocked_vlc:
        def set_media(self, media):
            pass

        def media_new(self, url):
            nonlocal vlc_media_url
            vlc_media_url = url
            return mocked_vlc_media()

        def play(self):
            nonlocal vlc_tried_to_start
            vlc_tried_to_start = True

        def stop(self):
            nonlocal vlc_tried_to_stop
            vlc_tried_to_stop = True

        def get_mrl(self):
            return None

        def media_player_new(self):
            return self  # is both, vlc instance and player

    def mock_vlc_inst() -> object:
        nonlocal vlc_tried_to_connect
        vlc_tried_to_connect = True
        return mocked_vlc()

    monkeypatch.setattr(cli.vlc, "Instance", mock_vlc_inst)

    # mock stream
    stream_url = None
    sound = AudioSegment.from_mp3(
        shared_datadir / "2020-05-26 15-47-05-113724_news.mp3"
    )
    output = BytesIO()
    sound.export(output, format="mp3")
    output.seek(0)

    class mock_stream:
        def read(self, length):
            return output.read(length)

    def mock_open_stream(stream, timeout):
        nonlocal stream_url
        stream_url = stream
        return mock_stream()

    monkeypatch.setattr(cli, "urlopen", mock_open_stream)

    # call cli
    runner = CliRunner()
    result = runner.invoke(cli.main, ["-su", "testuser", "--debug"])

    # spotify
    assert spot_tried_to_connect
    assert not spot_tried_to_start
    assert spot_tried_to_stop

    # vlc
    assert "nova" in vlc_media_url
    assert vlc_tried_to_connect
    assert vlc_tried_to_start
    assert not vlc_tried_to_stop

    # output
    assert "news" in result.output
    assert "switch to radio" in result.output
    assert "news" in result.output


def test_failing_spotify():
    pass


def test_non_existent_vlc():
    pass


def test_non_existent_stream():
    pass
