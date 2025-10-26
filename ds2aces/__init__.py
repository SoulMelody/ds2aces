import contextlib
import shutil

from loguru import logger

from ds2aces.ace.config import ace_studio_lock_file

__VERSION__ = "1.9.12"

def download_and_setup_ffmpeg():
    import portable_ffmpeg

    portable_ffmpeg.get_ffmpeg()
    portable_ffmpeg.add_to_path()

with contextlib.suppress(ImportError):
    if shutil.which("ffmpeg") is None:
        download_and_setup_ffmpeg()


if ace_studio_lock_file.exists():
    logger.error("ACE Studio is running. Please close it first.")
    raise SystemExit(1)