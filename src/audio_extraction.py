import logging
import os
import subprocess
from pathlib import Path
import json
from datetime import datetime,timedelta 
from argparse import Namespace
import glob 

def extract_audio(video_path, audio_directory):
    
    audio_path = video_path.split('/')[-1].replace('.mp4','.wav')
    audio_path = str(audio_directory / audio_path)
    args = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
        ]

    if os.path.isfile(audio_path):
        logging.info(f"{audio_path} already exists. Skipping.")
        return audio_path, None

    subprocess.call(args)
    logging.debug(f"Arguments: {args}")

    if not os.path.isfile(audio_path):
        msg = "File not created from ffmpeg operation %s" % audio_path
        logging.error(msg)
        raise IOError()
    return audio_path



if __name__ == "__main__":
    args = Namespace(**json.loads(os.getenv("args")))
    data_directory = Path(args.data_directory)
    session_id = args.session_id

    video_directory = Path(data_directory)  /  'videos'
    audio_directory = Path(data_directory) / 'audios'    
    video_directory.mkdir(parents=True, exist_ok=True)
    audio_directory.mkdir(parents=True, exist_ok=True)
    pattern = session_id + '*'
    for video_path in glob.glob(str(video_directory / pattern)):
        extract_audio(video_path, audio_directory)

