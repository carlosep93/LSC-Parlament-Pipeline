import json
import os
from argparse import Namespace
from pathlib import Path

from parlament_block_segmenter import ParlamentBlockSegmenter


def segment_parlament_data(data_directory: Path, session_id: str) -> None:
    alignment_data_path = (
        data_directory / "sessions" / session_id / f"output_{session_id}.json"
    )
    segments_directory = data_directory / "segments" / session_id

    segmenter = ParlamentBlockSegmenter(
        alignment_data_path, segments_directory, session_id
    )
    segmenter.segment_audio()


def main():
    args = Namespace(**json.loads(os.getenv("args")))

    data_directory = Path(args.data_directory)
    session_id = args.session_id

    segment_parlament_data(data_directory, session_id)


if __name__ == "__main__":
    main()
