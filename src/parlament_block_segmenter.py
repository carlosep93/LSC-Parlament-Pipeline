import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)


class ParlamentBlockSegmenter(object):
    def __init__(
        self, alignment_file: Path, segments_directory: Path, session_id: str
    ) -> None:
        self.session_id = session_id

        if os.path.isfile(alignment_file):
            self.alignment_file = alignment_file

            with open(alignment_file, "r", encoding="utf-8") as f:
                self.alignment_data = json.load(f)

            self.segments_directory = segments_directory
            os.makedirs(segments_directory, exist_ok=True)
        else:
            msg = f"Alignment file doesn't exist: {alignment_file}"
            raise IOError(msg)

    def segment_audio(self) -> None:
        metadata = []

        for cue in self.alignment_data:
            start = self.alignment_data[cue]["start"]
            end = self.alignment_data[cue]["end"]
            text = self.alignment_data[cue]["text"]
            speaker_meta = self.alignment_data[cue]["speaker_meta"]

            if "path_segment_audio" in self.alignment_data[cue]:
                path_audio = self.alignment_data[cue]["path_segment_audio"]
                path_audio = Path(path_audio)
                entry = self.segment_cue(
                    start, end, text, speaker_meta, path_audio=path_audio
                )
                metadata.append(entry)
            else:
                logging.info(f"'path_audio' key is missing in cue: {cue}. Skipping.")

        session_id_metadata_file = (
            self.segments_directory / f"{self.session_id}_metadata.json"
        )
        with open(session_id_metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    def segment_cue(
        self,
        start: str,
        end: str,
        text: str,
        speaker_meta: str,
        path_audio: Path = None,
    ) -> Dict[str, str]:
        if path_audio:
            return self._handle_path_audio(start, end, text, speaker_meta, path_audio)
        else:
            logging.warning("No valid audio source provided.")

    def _handle_path_audio(
        self,
        start: str,
        end: str,
        text: str,
        speaker_meta: str,
        path_audio: Path,
    ) -> Dict[str, str]:
        start_seconds = self._time_to_seconds(start)
        end_seconds = self._time_to_seconds(end)
        original_file = path_audio.stem

        duration = end_seconds - start_seconds
        audio_segment_file = (
            self.segments_directory
            / f"{original_file}_{start_seconds}_{end_seconds}.wav"
        )
        print(path_audio)
        if os.path.isfile(audio_segment_file):
            msg = f"{audio_segment_file} already exists, skipping"
            logging.info(msg)
        else:
            try:
                args = [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "panic",
                    "-i",
                    str(path_audio),
                    "-ss",
                    str(start_seconds),
                    "-t",
                    str(duration),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(audio_segment_file),
                ]
                print(' '.join(args))
                subprocess.call(args)
                logging.debug(f"Arguments: {args}")

                if not os.path.isfile(audio_segment_file):
                    msg = (
                        f"File not created from FFmpeg operation: {audio_segment_file}"
                    )
                    logging.warning(msg)
            except Exception as e:
                logging.error(f"Error during FFmpeg operation: {str(e)}")

        entry = self._create_audio_segment_metadata_entry(
            start=start,
            end=end,
            text=text,
            speaker_meta=speaker_meta,
            audio_segment_file=audio_segment_file,
            path_audio=path_audio,
        )

        return entry

    def _create_audio_segment_metadata_entry(
        self,
        start: str,
        end: str,
        text: str,
        speaker_meta: str,
        audio_segment_file: Path,
        path_audio: Path = None,
    ) -> Dict[str, str]:
        entry = {
            "audio_filepath": str(audio_segment_file),
            "text": text,
            "speaker_meta": speaker_meta,
            "start": start,
            "end": end,
        }

        if path_audio:
            entry["path_audio"] = str(path_audio)

        return entry

    def _time_to_seconds(self, time_str: str) -> int:
        h, m, s = map(int, time_str.split(":"))
        return h * 3600 + m * 60 + s
