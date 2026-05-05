import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Set

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)


def find_missing_and_empty_sessions(all_session_ids: Set, data_directory: Path):
    output_segments_dir = data_directory / "output_segment"
    final_output_dir = data_directory / "final_output"

    logging.info(f"Total number of sessions: {len(all_session_ids)}\n")

    output_segment_ids = set()
    empty_output_segment_dirs = set()
    for dir in os.listdir(output_segments_dir):
        output_segment_ids.add(dir)
        if not os.listdir(output_segments_dir / dir):
            empty_output_segment_dirs.add(dir)

    missing_output_segment_sessions = all_session_ids - output_segment_ids
    logging.info(
        f"Total number of MISSING output segment sessions: {len(missing_output_segment_sessions)}\n"
    )
    logging.info(
        f"Percentage of MISSING output segment sessions: {(len(missing_output_segment_sessions) / len(all_session_ids)) * 100}%\n"
    )
    logging.info(
        f"All MISSING output segment sessions: {missing_output_segment_sessions}\n"
    )

    final_output_ids = set()
    empty_final_output_dirs = set()
    for dir in os.listdir(final_output_dir):
        final_output_ids.add(dir)
        if not os.listdir(final_output_dir / dir):
            empty_final_output_dirs.add(dir)

    missing_final_output_sessions = all_session_ids - final_output_ids
    logging.info(
        f"Total number of MISSING final output sessions: {len(missing_final_output_sessions)}\n"
    )
    logging.info(
        f"Percentage of MISSING final output sessions: {(len(missing_final_output_sessions) / len(all_session_ids)) * 100}%\n"
    )
    logging.info(
        f"All MISSING final output sessions: {missing_final_output_sessions}\n"
    )

    logging.info(
        f"Total number of EMPTY folders in {final_output_dir}: {len(empty_final_output_dirs)}\n"
    )
    logging.info(f"All EMPTY final output session folders: {empty_final_output_dirs}\n")
    logging.info(
        f"Total number of MISSING or EMPTY final output session folders: {len(missing_final_output_sessions) + len(empty_final_output_dirs)}\n"
    )

    logging.info(
        f"Total number of EMPTY folders in {output_segments_dir}: {len(empty_output_segment_dirs)}\n"
    )
    logging.info(f"All EMPTY output segment folders: {empty_output_segment_dirs}\n")
    logging.info(
        f"Total number of MISSING or EMPTY output segment folders: {len(missing_output_segment_sessions) + len(empty_output_segment_dirs)}\n"
    )

    logging.info(
        f"Empty 'final_output' folders that ARE NOT missing from 'output_segment/' dir: {empty_final_output_dirs - missing_output_segment_sessions}\n"
    )


def find_parlament_missing_sessions(data_directory: Path):
    audios_dir = data_directory / "segments"

    all_sessions_ids = set()
    for dir in os.listdir(audios_dir):
        all_sessions_ids.add(dir)

    find_missing_and_empty_sessions(all_sessions_ids, data_directory)


def find_corts_missing_sessions(data_directory: Path):
    audios_dir = data_directory / "audios"

    all_session_ids = set()
    for file in os.listdir(audios_dir):
        session_id = "_".join(file.split("_")[:-1])
        all_session_ids.add(session_id)

    find_missing_and_empty_sessions(all_session_ids, data_directory)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--pipeline",
        "-p",
        choices=["corts", "parlament"],
        type=str.lower,
        help="The name of the pipeline you wish to run",
        required=True,
    )
    parser.add_argument(
        "--data-directory",
        "-d",
        help="The relative path to the pipeline data directory",
        required=True,
    )

    args = parser.parse_args()
    data_directory = Path(args.data_directory)

    if args.pipeline == "corts":
        find_corts_missing_sessions(data_directory)
    elif args.pipeline == "parlament":
        find_parlament_missing_sessions(data_directory)


if __name__ == "__main__":
    main()
