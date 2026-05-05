import json
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

from parlament_alignment import ParlamentAlignment

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)


def extract_diari_files_and_datetimes_from_corts_diaris_metadata(
    diaris_metadata: Dict,
) -> Tuple[List[str], List[str]]:
    diaris_files_list = diaris_metadata["diari_files"]
    dates = diaris_metadata["diari_datetimes"]
    return diaris_files_list, dates


def extract_session_name_from_metadata_file_name(metadata_file_path: Path) -> str:
    return metadata_file_path.stem


def extract_entries_from_metadata_file(metadata_file_path: Path) -> List[Dict]:
    with open(metadata_file_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    entries = metadata["data"]["record"]["playlist"]["entrys"]
    return entries


def write_transcriptions_and_metadata_to_temp_file(
    data_directory: Path,
    all_data: Dict,
    session_id: str,
) -> None:
    temp_output_file = data_directory / f"temp_{session_id}.json"
    if not temp_output_file.is_file():
        with open(temp_output_file, "w", encoding="utf-8") as metadata_file:
            json.dump(all_data, metadata_file, ensure_ascii=True, indent=4)
    else:
        with open(temp_output_file, "r", encoding="utf-8") as metadata_file:
            alignment_data = json.load(metadata_file)
        alignment_data.update(all_data)
        with open(temp_output_file, "w", encoding="utf-8") as metadata_file:
            json.dump(alignment_data, metadata_file, ensure_ascii=True, indent=4)


def _create_parlament_block_alignment(
    session_id: str,
    data_directory: Path,
):
    align = ParlamentAlignment("parlament", data_directory, session_id)
    align.block_align()
    align.output_media_vs_text()


def perform_parlament_block_alignment(data_directory: Path, session_id: str):
    output_dir = data_directory / "sessions"
    os.makedirs(output_dir, exist_ok=True)

    _create_parlament_block_alignment(session_id, data_directory)


def main():
    args = Namespace(**json.loads(os.getenv("args")))
    print(args)
    data_directory = Path(args.data_directory)
    session_id = args.session_id

    perform_parlament_block_alignment(data_directory, session_id)


if __name__ == "__main__":
    main()
