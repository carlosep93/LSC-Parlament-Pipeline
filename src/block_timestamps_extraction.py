import json
import logging
import os
import re
import subprocess
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Union


def perform_nemo_alignment(
    session_output_dir: Path,
    manifest_file_path: Path,
    pretrained_name: str,
    ca_nemo_asr_model_path: Union[Path, None] = None,
):
    command = [
        "python3",
        "NeMo/tools/nemo_forced_aligner/align.py",
        f"manifest_filepath={str(manifest_file_path)}",
        f"output_dir={str(session_output_dir)}",
        "align_using_pred_text=false",
        "transcribe_device=cpu",
        "viterbi_device=cpu",
        "additional_segment_grouping_separator=*",
    ]

    if ca_nemo_asr_model_path:
        command.append(f"model_path={ca_nemo_asr_model_path}")
    else:
        command.append(f"pretrained_name={pretrained_name}")

    subprocess.run(command, shell=False)


def clean_text(input_text):
    """Replace '.' in numbers with '' to avoid FA segmentation errors.
    Remove all tags between (), such as (Aplaudiments).
    Remove special chararters in 'remove_chars' list.
    """
    # output_text = re.sub(r'(\d+)\. (\d+)', r'\1\2', input_text) # assuming a space after the dot (18. 000 -> 18000) #this for corts
    output_text = re.sub(
        r"(\d+)\.(\d+)", r"\1\2", input_text
    )  # 18.000 -> 18000 #this for parlament
    output_text = re.sub(r"(\d+)\,(\d+)", r"\1 coma \2", output_text)

    output_text = re.sub(r"\([^)]*\)", "", output_text)  # removing (tags)
    output_text = re.sub(
        r"(\d+)/(\d+)", r"\1 \2", output_text
    )  # Llei 16/1991 -> 16 1991
    output_text = output_text.replace("...", ".")

    remove_chars = [
        ",",
        ";",
        ":",
        "¿",
        "?",
        "«",
        "»",
        "-",
        "¡",
        "!",
        "@",
        ".",
        "{",
        "}",
        "[",
        "]",
        "=",
        "/",
        "\\",
        "&",
        "#",
        "…",
    ]
    # We dont remove * because we use them in the segmentation

    output_text = "".join(
        char if char not in remove_chars else " " for char in output_text
    )  # removing special chars
    return " ".join(
        output_text.split()
    ).lower()  # Remove extra spaces and return cleaned text


def output_to_manifest_block(
    blocks_file_path: Path, data_directory: Path, session_id: str
):
    with open(blocks_file_path, "r", encoding="utf-8") as metadata_file:
        data = json.load(metadata_file)

    manifest_dir = data_directory / "manifest_block"
    os.makedirs(manifest_dir, exist_ok=True)

    manifest_file_path = manifest_dir / f"{session_id}_manifest.json"
    with open(manifest_file_path, "w", encoding="utf-8") as f:
        text_accumulator = []
        for value in data:
            text = "".join(value["text"])
            processed_text = clean_text(text)
            text_accumulator.append(processed_text)

        joined_text = " ".join(text_accumulator)
        if joined_text:
            nemo_manifest = {
                "audio_filepath": f"data/balear/audios/{session_id}.flac",
                "text": joined_text,
            }
            json.dump(nemo_manifest, f)
            f.write("\n")

    return manifest_file_path


def main():
    args = Namespace(**json.loads(os.getenv("args")))
    session_id = args.session_id
    data_directory = Path(args.data_directory)
    logs_dir = data_directory / "logs"
    os.makedirs(logs_dir, exist_ok=True)

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    log_file = logs_dir / f"block_output_{session_id}_{date_time}.log"
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s]: %(message)s",
        level=logging.INFO,
        filename=str(log_file),
        filemode="w",
    )

    metadata_path_block = data_directory / "jsons"
    if not os.path.isdir(metadata_path_block):
        raise IOError("JSON directory not found!")

    ca_nemo_asr_model_path = Path(args.ca_nemo_model) if args.ca_nemo_model else None
    pretrained_name = "stt_ca_conformer_ctc_large"

    logging.info("Nemo Alignment\n")

    output_dir = data_directory / "blocklevel_alignment"
    os.makedirs(output_dir, exist_ok=True)

    output_path_final_json = output_dir / f"final_output_{session_id}.json"
    with open(output_path_final_json, "w", encoding="utf-8") as final_output:
        for root, _, files in os.walk(metadata_path_block):
            for blocks_file in files:
                if blocks_file.endswith("_speaker.json") and session_id in blocks_file:
                    logging.info(f"Blocks file: {str(blocks_file)}")
                    blocks_file_path = Path(root) / blocks_file
                    start_time_manifest = time.perf_counter()

                    logging.info("Generate Manifests:")
                    manifest_file_path = output_to_manifest_block(
                        blocks_file_path, data_directory, session_id
                    )
                    end_time_manifest = time.perf_counter()
                    t = end_time_manifest - start_time_manifest
                    logging.info(f"Manifest Creation time: {t}")

                    logging.info("Perform NEMO Alignment:")
                    logging.info(f"Start NFA with: {str(manifest_file_path)}")
                    session_output_dir = output_dir / session_id
                    perform_nemo_alignment(
                        session_output_dir,
                        manifest_file_path,
                        pretrained_name,
                        ca_nemo_asr_model_path,
                    )
                    logging.info(f"End NFA with: {str(manifest_file_path)}\n")


if __name__ == "__main__":
    main()
