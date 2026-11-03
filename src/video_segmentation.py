import logging
import os
import subprocess
from pathlib import Path
import json
from datetime import datetime,timedelta 
from argparse import Namespace


def load_metadata(metadata_path):
    with open(metadata_path) as f:
        metadata = json.load(f)
    return metadata


def get_base_name(name):
    base_name = name.split('/')[-1]
    base_name = base_name.split('.')[0]
    base_name = base_name.split('_')
    if len(base_name) == 4:
        return '_'.join(base_name[:2])
    return base_name[0]

def segment_video(data_directory, segment, out_path):
    
    base_name = get_base_name(segment['audio_filepath'])
    video_file = '.'.join([base_name, 'mp4'])
    video = os.path.join(data_directory, 'videos', video_file)

    start = datetime.strptime(segment['start'], '%H:%M:%S').time()
    start_seconds = timedelta(hours=start.hour,minutes=start.minute,seconds=start.second).total_seconds()
    start = datetime.combine(datetime.now(), start)
    end = datetime.strptime(segment['end'], '%H:%M:%S').time()
    end_seconds = timedelta(hours=end.hour,minutes=end.minute,seconds=end.second).total_seconds()
    end = datetime.combine(datetime.now(), end)
    if end <= start:
        # blocks spanning two video files get end < start from the metadata; ffmpeg can't cut them
        logging.warning(f"Skipping {base_name} {segment['start']}-{segment['end']}: end <= start")
        return None
    duration = str(end - start)

    file_name = "_".join([base_name, str(int(start_seconds)), str(int(end_seconds))]) + ".mp4"
    file_path = os.path.join(out_path, file_name)
     
    args = [
        "ffmpeg",
        "-ss",
        str(start).split()[-1],
        "-i",
        video,
        "-t",
        duration,
        "-vf",
        "crop=330:330:30:75",
        "-vsync",
        "0",
        file_path,
        ]

    if os.path.isfile(file_path):
        logging.info(f"{file_path} already exists. Skipping.")
        return file_path, None

    subprocess.call(args)
    logging.debug(f"Arguments: {args}")

    if not os.path.isfile(file_path):
        msg = "File not created from ffmpeg operation %s" % file_path
        logging.error(msg)
        raise IOError()
    return file_path



if __name__ == "__main__":
    args = Namespace(**json.loads(os.getenv("args")))
    data_directory = Path(args.data_directory)
    session_id = args.session_id
    metadata_file = session_id + '_metadata.json'
    metadata_path = os.path.join(data_directory, "segments", session_id, metadata_file)
    metadata = load_metadata(metadata_path)
    
    output_path = Path(os.path.join(data_directory,'video_segments', session_id))
    output_path.mkdir(parents=True, exist_ok=True)

    for segment in metadata:
        segment_video(data_directory, segment, output_path)
             



