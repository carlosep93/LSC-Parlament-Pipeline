import json
import logging
import os
import re
import shutil
import subprocess
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict

import fasttext
import inflect
import nemo.collections.asr as nemo_asr
from dotenv import dotenv_values
from fasttext.FastText import _FastText
from minio import Minio
from s3path import S3Path

from asr import AudioSpeechRecognition
from databases import Database, MongoDatabase
from segmenter import Segmenter

config = dotenv_values(".env")


class DatasetGenerator:
    def __init__(
        self,
        pipeline: str,
        data_directory: Path,
        remote_storage: bool,
        database: Database,
        client: Minio,
        session_id: str,
    ) -> None:
        self.pipeline = pipeline
        self.data_directory = data_directory
        self.database = database
        self.client = client
        self.remote_storage = remote_storage
        self.s3_bucket = (
            S3Path(config[f"MINIO_{self.pipeline.upper()}_BUCKET_NAME"])
            if self.remote_storage
            else None
        )
        self.session_id = session_id
        self.audio_files, self.video_files = self.__load_audio_video_files()

    def __clean_text(self, input_text: str, lang: str):
        """Replace '.' in numbers with '' to avoid FA segmentation errors.
        Remove all tags between (), such as (Aplaudiments).
        Remove special chararters in 'remove_chars' list.
        """

        if self.pipeline in ["balear", "ib3", "parlament"]:
            output_text = re.sub(
                r"(\d+)\.(\d+)", r"\1\2", input_text
            )  # 18.000 -> 18000 this for parlament
        elif self.pipeline == "corts":
            output_text = re.sub(
                r"(\d+)\. (\d+)", r"\1\2", input_text
            )  # assuming a space after the dot (18. 000 -> 18000)

        output_text = re.sub(r"(\d+)\,(\d+)", r"\1 coma \2", output_text)
        output_text = re.sub(r"\([^)]*\)", "", output_text)  # removing (tags)
        output_text = re.sub(
            r"(\d+)/(\d+)", r"\1 \2", output_text
        )  # Llei 16/1991 -> 16 1991
        output_text = output_text.replace("...", ".")

        if self.pipeline == "balear":
            output_text = self.__balear_replace_abbreviations_in_text(lang, output_text)

        pattern_percent = r"%"
        if self.pipeline == "corts" or self.pipeline == "parlament":
            if lang == "__label__ca":
                output_text = re.sub(pattern_percent, " per cent", output_text)
                #output_text = self.__numbers_to_chars(
                #    output_text, "cat"
                #)  # Converting Catalan numbers to text
            elif lang == "__label__es":
                output_text = re.sub(pattern_percent, " por ciento", output_text)
                #output_text = self.__numbers_to_chars(output_text, "spa")
        else:
            if lang == "__label__ca":
                output_text = re.sub(pattern_percent, " per cent", output_text)
                #output_text = self.__balear_ib3_numbers_to_chars(
                #    output_text, "cat"
                #)  # Converting Catalan numbers to text
            elif lang == "__label__es":
                output_text = re.sub(pattern_percent, " por ciento", output_text)
                #output_text = self.__balear_ib3_numbers_to_chars(
                #    output_text, "spa"
                #)  # Converting Spanish numbers to text

        remove_chars = [
            ",",
            ";",
            ":",
            "¿",
            "?",
            "«",
            "»",
            "¡",
            "!",
            "@",
            "*",
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

        if self.pipeline == "ib3":
            remove_chars.extend(['"', "'", "."])
        else:
            remove_chars.append("-")

        output_text = "".join(
            char if char not in remove_chars else " " for char in output_text
        )  # removing special chars
        return " ".join(
            output_text.split()
        ).lower()  # Remove extra spaces and return cleaned text

    def __balear_replace_abbreviations_in_text(
        self, lang: str, output_text: str
    ) -> str:
        if lang == "__label__ca":
            output_text = re.sub(
                r"\bsra\b\.", "senyora", output_text, flags=re.IGNORECASE
            )
            output_text = re.sub(
                r"\bSra\b\.", "senyora", output_text, flags=re.IGNORECASE
            )
            output_text = re.sub(
                r"\bsr\b\.", "senyor", output_text, flags=re.IGNORECASE
            )
            output_text = re.sub(
                r"\bSr\b\.", "senyor", output_text, flags=re.IGNORECASE
            )
        elif lang == "__label__es":
            output_text = re.sub(
                r"\bsra\b\.", "señora", output_text, flags=re.IGNORECASE
            )
            output_text = re.sub(
                r"\bSra\b\.", "señora", output_text, flags=re.IGNORECASE
            )
            output_text = re.sub(r"\bsr\b\.", "señor", output_text, flags=re.IGNORECASE)
            output_text = re.sub(r"\bSr\b\.", "señor", output_text, flags=re.IGNORECASE)

        return output_text

    def __translate(self, text, destination_language):
        # Check if the destination language is valid
        if destination_language not in ["spa", "cat"]:
            raise IOError(
                f"Invalid destination language: {destination_language}. \
                            Supported languages are 'spa' or 'cat'."
            )

        command = f'echo "{text}" | apertium eng-{destination_language}'
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output, error = process.communicate()

        # Check if there was an error during execution
        if error:
            raise IOError(f"Error: {error}")
        else:
            return output.strip()

    def __numbers_to_chars(self, text: str, lang: str):
        p = inflect.engine()
        matches = re.finditer(r"\b\d+\b", text)
        modified_parts = []
        last_index = 0

        for match in matches:
            start, end = match.start(), match.end()
            before = text[last_index:start]
            number = text[start:end]
            number_as_word = p.number_to_words(int(number))
            translated_text = self.__translate(number_as_word, lang)
            modified_parts.extend([before, translated_text])
            last_index = end
        modified_parts.append(text[last_index:])
        return "".join(modified_parts)

    def __get_text_and_speaker(self, audio_segment_data: Dict):
        match self.pipeline:
            case "parlament":
                text = " ".join(
                    e[1] for e in audio_segment_data["text"] if "mesa" not in e[0]
                )
                # not to accumulate speaker's name when more interventions are in the same block
                speakers = set(
                    e[0] for e in audio_segment_data["text"] if "mesa" not in e[0]
                )
                speaker = "".join(speakers)
            case "corts":
                text = ""
                speaker = ""
                for e in audio_segment_data["text"]:
                    if "president" not in e[0]:
                        text = f"{text} {e[1]}"
                        speaker = f"{speaker};{e[0]}"
            case "balear":
                speaker = audio_segment_data["speaker"]
                text = audio_segment_data["text"]

        return text, speaker

    def __write_nemo_manifest_to_file(
        self, audio_segment_data: Dict, out, lgid_model: _FastText
    ):
        text, speaker = self.__get_text_and_speaker(audio_segment_data)

        if text and speaker:
            (lang, conf) = AudioSpeechRecognition.identify_language_with_confidence(
                lgid_model, text
            )
            language = lang + "__" + str("{:.2f}".format(conf))

            processed_text = self.__clean_text(text, lang)
            nemo_manifest = {
                "audio_filepath": audio_segment_data["audio_filepath"],
                "language": language,
                "speaker": speaker,
                "text": processed_text,
            }
            json.dump(nemo_manifest, out)
            out.write("\n")

    def __output_to_manifest(self, metadata: Dict, lgid_model: _FastText):
        manifest_dir = self.data_directory / "manifest"
        os.makedirs(manifest_dir, exist_ok=True)

        manifest_file_path = manifest_dir / f"{self.session_id}_manifest.json"
        with open(manifest_file_path, "w", encoding="utf-8") as out:
            for audio_segment_data in metadata:
                if audio_segment_data:
                    if self.pipeline in ["balear", "ib3", "parlament"]:
                        can_dump_json = (
                            audio_segment_data["start"] < audio_segment_data["end"]
                        )
                    else:
                        can_dump_json = float(audio_segment_data["start"]) < float(
                            audio_segment_data["end"]
                        )

                    if can_dump_json:
                        if self.pipeline == "ib3":
                            text = audio_segment_data.get("text")
                            if text:
                                lang, conf = self.__identify_language(text, lgid_model)
                                language = lang + "__" + str("{:.2f}".format(conf))
                                processed_text = self.__clean_text(text, lang)
                                nemo_manifest = {
                                    "audio_filepath": audio_segment_data[
                                        "audio_filepath"
                                    ],
                                    "language": language,
                                    "text": processed_text,
                                }
                            json.dump(nemo_manifest, out)
                            out.write("\n")
                        else:
                            self.__write_nemo_manifest_to_file(
                                audio_segment_data, out, lgid_model
                            )

        return manifest_file_path

    def __identify_language(self, text, model):
        predictions = model.predict(text, k=1)
        lang_id = predictions[0][0]
        confidence = predictions[1][0]
        return lang_id, confidence

    def __setup(
        self, ca_nemo_asr_model_path: Path = None, es_nemo_asr_model_path: Path = None
    ):
        # Paths
        segments_path: Path = data_directory / "segments" / self.session_id
        os.makedirs(segments_path, exist_ok=True)

        # Language Identification:
        if self.pipeline == "corts" or self.pipeline == "parlament":
            lgid_model_path: str = "data/models/lid.176.ftz"
        else:
            lgid_model_path: str = "data/models/lid.176.bin"
        lgid_model: _FastText = fasttext.load_model(lgid_model_path)

        # NFA:
        pretrained_name: str = "stt_ca_conformer_ctc_large"

        # ASR:
        if ca_nemo_asr_model_path and es_nemo_asr_model_path:
            ca_asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
                restore_path=str(ca_nemo_asr_model_path)
            )
            es_asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
                restore_path=str(es_nemo_asr_model_path)
            )
        else:
            ca_asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                model_name="stt_ca_conformer_ctc_large"
            )
            es_asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                model_name="stt_es_conformer_ctc_large"
            )

        return (segments_path, lgid_model, pretrained_name, ca_asr_model, es_asr_model)

    def __generate_manifest(self, block_data: Dict, lgid_model: _FastText):
        start_time_manifest = time.perf_counter()
        logging.info("1. Generate Manifests:\n")

        manifest_file_path = self.__output_to_manifest(block_data, lgid_model)

        end_time_manifest = time.perf_counter()
        t = end_time_manifest - start_time_manifest
        logging.info(f"Manifest Creation time: {t}\n")
        return manifest_file_path

    def __perform_nemo_alignment(
        self,
        manifest_file_path,
        output_dir,
        pretrained_name,
        ca_nemo_asr_model_path=None,
    ):
        logging.info("2. Perform NEMO Alignment:\n")
        logging.info(f"Output file: {output_dir}\n")
        logging.info(f"Start NFA with: {manifest_file_path}\n")

        command = [
            "python3",
            "NeMo/tools/nemo_forced_aligner/align.py",
            f"manifest_filepath={manifest_file_path}",
            f"output_dir={output_dir}",
            "align_using_pred_text=false",
            "transcribe_device=cpu",
            "viterbi_device=cpu",
        ]

        if ca_nemo_asr_model_path:
            command.append(f"model_path={ca_nemo_asr_model_path}")
        else:
            command.append(f"pretrained_name={pretrained_name}")

        if self.pipeline in ["balear", "corts", "parlament"]:
            command.append("additional_segment_grouping_separator=.")
        
        subprocess.run(command, shell=False)
        logging.info(f"End NFA with: {manifest_file_path}\n")

    def __create_out_dir(self):
        out_dir = data_directory / f"output_segment/{self.session_id}"
        os.makedirs(out_dir, exist_ok=True)
        s3_out_path = S3Path("output_segment", self.session_id)
        return out_dir, s3_out_path


    def __segmentation_and_scoring(
        self,
        output_dir: Path,
        lgid_model: _FastText,
        ca_asr_model,
        es_asr_model,
        block_data: Dict,
    ):
        logging.info(f"3. Start Segmentation and Scoring: \n")
        alignment_dir = output_dir / "ctm" / "segments"
        session_id_final_output_dir = (
            self.data_directory / f"final_output/{self.session_id}"
        )
        os.makedirs(alignment_dir, exist_ok=True)
        os.makedirs(session_id_final_output_dir, exist_ok=True)

        start_time_transcription = time.perf_counter()
        for root, _, files in os.walk(alignment_dir):
            for index, alignment_block_file in enumerate(files):
                block_id = (
                    alignment_block_file.split(".")[0]
                )
                block_audio_file_path = data_directory / "segments"
                block_video_file_path = data_directory / "video_segments" 
                
                block_audio_file = (
                    block_audio_file_path / self.session_id / f"{block_id}.wav"
                )

                block_video_file = (
                    block_video_file_path / self.session_id / f"{block_id}.mp4"
                )
                out_path, s3_out_path = self.__create_out_dir()

                try:
                    segmenter = Segmenter(
                        root + "/" + alignment_block_file,
                        str(block_audio_file),
                        str(block_video_file),
                        out_path,
                        lgid_model,
                        ca_asr_model,
                        es_asr_model,
                        self.pipeline,
                        self.s3_bucket,
                        s3_out_path,
                        self.client,
                        self.data_directory,
                        self.remote_storage,
                    )
                    
                    desired_block = next(
                        item
                        for item in block_data
                        if item and block_id in item["audio_filepath"]
                    )

                    speaker = desired_block.get("speaker_meta")
                    
                    audio_results_segmentation = segmenter.segment_audio(
                        block_id, speaker, self.audio_files
                    )
                    logging.info(audio_results_segmentation)
                    video_results_segmentation = segmenter.segment_video(
                        block_id, audio_results_segmentation, self.video_files
                    )
                    logging.info(video_results_segmentation)
                    block_dict = {
                        "block_id": block_id,
                        "block_data": desired_block,
                        "results": video_results_segmentation
                    }

                    self.__store_results(
                        block_dict, block_id, index, session_id_final_output_dir
                    )
                except IOError as e:
                    logging.error(e)

        if self.remote_storage:
            rm_path = data_directory / "output_segment"
            if rm_path.exists():
                shutil.rmtree(str(rm_path))

        end_time_transcription = time.perf_counter()
        t = end_time_transcription - start_time_transcription
        logging.info(f"Segmentation and Scoring time: {t}\n")

    def __store_results(
        self,
        block_dict: dict,
        block_id: str,
        index: int,
        final_output_dir: Path,
    ):
        if self.remote_storage:
            try:
                if not self.database.find("block_id", block_id):
                    self.database.insert(block_dict)
            except ValueError as e:
                logging.error(f"Could not store data in database. {e}")
        else:
            final_output_file = (
                final_output_dir / f"{self.session_id}_final_output_{index}.json"
            )
            with open(final_output_file, "w", encoding="utf-8") as f:
                json.dump(block_dict, f, indent=4)

    def __get_most_common_speaker(self, speakers_list: list) -> str:
        not_allowed_speakers = ["president", "presidenta", "mesa"]
        speaker_frequency = {}

        for speaker in speakers_list:
            if speaker[0] not in not_allowed_speakers:
                speaker_frequency[speaker[0]] = speaker_frequency.get(speaker[0], 0) + 1

        if len(speaker_frequency) == 0:
            return "unknown"

        if len(speaker_frequency) > 1:
            logging.warning("There is more than one speaker.")

        return max(speaker_frequency, key=lambda speaker: speaker_frequency[speaker])

    def generate_final_data(
        self, ca_nemo_asr_model_path: Path = None, es_nemo_asr_model_path: Path = None
    ):
        
        (
            segments_path,
            lgid_model,
            pretrained_name,
            ca_asr_model,
            es_asr_model,
        ) = self.__setup(ca_nemo_asr_model_path, es_nemo_asr_model_path)
    
        # Nemo Alignment
        logging.info("Nemo Alignment\n")
        start_time_total = time.perf_counter()
        output_directory = data_directory / "wordlevel_alignment"
        os.makedirs(output_directory, exist_ok=True)

        if self.pipeline in ["corts", "parlament", "balear"]:
            end_of_file_name = "_metadata.json"
        else:
            end_of_file_name = "_segments.json"
        
        for file in os.listdir(segments_path):
            if file.endswith(end_of_file_name):
                wordlevel_alignment_directory = output_directory / self.session_id
                #if not os.path.exists(wordlevel_alignment_directory):
                if True:
                    logging.info(f"Blocks file: {file}")
                    blocks_file_path = segments_path / file
                    with open(blocks_file_path, "r", encoding="utf-8") as f:
                        block_data = json.load(f)
                    
                    manifest_file_path = self.__generate_manifest(
                        block_data, lgid_model
                    )
                    
                    output_dir = output_directory / self.session_id
                    
                    # REMOVE COMMENT
                    self.__perform_nemo_alignment(
                        manifest_file_path,
                        output_dir,
                        pretrained_name,
                        ca_nemo_asr_model_path,
                    )
                
                    self.__segmentation_and_scoring(
                        output_dir,
                        lgid_model,
                        ca_asr_model,
                        es_asr_model,
                        block_data,
                    )

        end_time_total = time.perf_counter()
        t = end_time_total - start_time_total
        logging.info(f"Total Execution time: {t}\n")


    def __fill_files_dict(self, files_dir):
        files_dict = {}
        files = [file for file in os.listdir(files_dir) if self.session_id in file]
        for file in files:
            self.__update_files_dict(files_dict, file)

        if self.pipeline == "parlament" and not files_dict:
            file_dir = self.data_directory / "segments" / self.session_id
            files = [
                file
                for file in os.listdir(file_dir)
                if "metadata.json" not in file
            ]
            for file in files:
                self.__update_files_dict(files_dict, file)
        return files_dict

    def __load_audio_video_files(self):
        audio_files = {}
        video_files = {}
        audio_dir = self.data_directory / "audios"
        video_dir = self.data_directory / "videos"
        audio_files = self.__fill_files_dict(audio_dir)
        video_files = self.__fill_files_dict(video_dir)
        return audio_files, video_files

    def __update_files_dict(self, audio_files: dict, file: str):
        file_and_extension = file.split(".")
        file = file_and_extension[0]
        extension = file_and_extension[1]
        audio_files.update({file: extension})


if __name__ == "__main__":
    args = Namespace(**json.loads(os.getenv("args")))
    data_directory = Path(args.data_directory)
    database = MongoDatabase("parlament") if args.remote_storage else None
    client = (
        Minio(
            endpoint=config["MINIO_ENDPOINT"],
            access_key=config["MINIO_ACCESS_KEY"],
            secret_key=config["MINIO_SECRET_KEY"],
        )
        if args.remote_storage
        else None
    )
    pipeline = "parlament"
    session_id = args.session_id
    logs_dir = Path(f"data/logs/{pipeline}")
    os.makedirs(logs_dir, exist_ok=True)

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    log_file = logs_dir / f"output_{session_id}_{date_time}.log"
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s]: %(message)s",
        level=logging.INFO,
        filename=str(log_file),
        filemode="w",
    )

    if args.ca_nemo_model:
        ca_nemo_asr_model_path = Path(args.ca_nemo_model)
    else:
        ca_nemo_asr_model_path = None

    if args.es_nemo_model:
        es_nemo_asr_model_path = Path(args.es_nemo_model)
    else:
        es_nemo_asr_model_path = None
    
    logging.info('Logging works')
    
    dataset_generator = DatasetGenerator(
        pipeline,
        data_directory,
        args.remote_storage,
        database,
        client,
        session_id,
    )
    dataset_generator.generate_final_data(
        ca_nemo_asr_model_path, es_nemo_asr_model_path
    )