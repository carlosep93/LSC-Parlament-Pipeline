import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Tuple
from datetime import datetime,timedelta 

import pandas as pd
from jiwer import cer
from minio import Minio
from pyctcdecode import build_ctcdecoder
from s3path import S3Path

from asr import AudioSpeechRecognition

#Import sign language detection module (not good)
# from sign_detector import SignDetector

from sign_audio_transition_detection import CustomSignSegmentation


class Segmenter(object):
    def __init__(
        self,
        alignment_file: str,
        audio_file: str,
        video_file: str,
        out_path: Path,
        lg_model: Any,
        ca_model: Any,
        es_model: Any,
        pipeline: str,
        s3_bucket: S3Path,
        s3_out_path: S3Path,
        minio_client: Minio,
        data_directory: Path,
        remote_storage: bool,
    ):
        if os.path.isfile(alignment_file) and os.path.isfile(audio_file):
            self.video_file = video_file
            self.audio_file = audio_file
            logging.info('Alignment file:')
            logging.info(alignment_file)
            self.alignment = pd.read_csv(
                alignment_file,
                sep=" ",
                names=["file", "nb", "start", "duration", "text"],
            )
            self.out_path = out_path
            self.lg_model = lg_model
            self.ca_model = ca_model
            self.es_model = es_model
            self.pipeline = pipeline
            self.s3_bucket = s3_bucket
            self.s3_out_path = s3_out_path
            self.minio_client = minio_client
            self.data_directory = data_directory
            self.remote_storage = remote_storage
            if not os.path.isdir(out_path):
                os.makedirs(out_path)
        else:
            msg = "audio file or alignment file doesn't exist:" "\n  %s\n  %s" % (
                audio_file,
                alignment_file,
            )
            raise IOError(msg)

    def __get_original_audio_path(
        self, audio_file_name: str, audio_file_extension: str
    ) -> str:
        if not audio_file_extension:
            return None

        source_dir = self.s3_bucket if self.remote_storage else self.data_directory
        if self.pipeline == "parlament" and audio_file_extension == "wav":
            session_id = audio_file_name.split("_")[0]
            original_audio_path = str(
                source_dir
                / "segments"
                / f"{session_id}"
                / f"{audio_file_name}.{audio_file_extension}"
            )
        else:
            original_audio_path = str(
                source_dir / "audios" / f"{audio_file_name}.{audio_file_extension}"
            )

        return original_audio_path


    def __get_base_name(self,name):
        base_name = name.split('/')[-1]
        base_name = base_name.split('.')[0]
        base_name = base_name.split('_')
        if len(base_name) == 4:
            return '_'.join(base_name[:2])
        return base_name[0]

    # Currently not being used
    def __align_audio_and_video_old(self, audio_result, transitions, margin=4):
        audio_start = audio_result['start']
        audio_end = audio_result['end']
        sign_start = audio_start
        sign_end = audio_end + margin
        for sign in transitions:
            if sign.start-margin <=  audio_start <= sign.start+margin and (sign.end - audio_end) < margin*3:
                    sign_start = min(audio_start,sign.start)
                    sign_end = max(audio_end, sign.end)
                    break
        return sign_start, sign_end, audio_start, audio_end
    
    def __pair_audio_video_segments(self, audio_segments, video_segments, margin=4, threshold=0.3):
        pairs = []
        missing = 0
        for a_s in audio_segments:
            video_start = -1
            video_end = -1
            for v_s in video_segments:
                if  video_start == -1 and  a_s[0] <=  v_s[0] <= a_s[1]:
                    video_start = v_s[0]

                if  not video_start == -1 and video_end == -1 and a_s[1] <= v_s[1] <= a_s[1] + margin and v_s[2] >= threshold:
                    video_end = v_s[1]

                if  not video_start == -1 and \
                    not video_end == -1:
                    pairs.append({
                        'audio_start': a_s[0],
                        'audio_end': a_s[1],
                        'video_start': video_start,
                        'video_end': video_end
                        })
                    break
            if video_start == -1 or video_end == -1:
                missing += 1
                pairs.append({
                    'audio_start': a_s[0],
                    'audio_end': a_s[1],
                    'video_start': a_s[0],
                    'video_end': a_s[1] + margin
                })
        return pairs

    def __align_audio_and_video(self, audio_start, audio_end, video_segments, margin=4, threshold=0.3):
        video_start = -1
        video_end = -1
        for v_s in video_segments:
            if  video_start == -1 and  audio_start <= v_s[0] <= audio_end:
                video_start = v_s[0]

            if  not video_start == -1 and video_end == -1 and audio_end <= v_s[1] <= audio_end + margin and v_s[2] >= threshold:
                video_end = v_s[1]

            if  not video_start == -1 and not video_end == -1:
                return video_start, video_end, audio_start, audio_end
                break

            if video_start == -1 or video_end == -1:
                return audio_start, audio_end + margin, audio_start, audio_end

            # print('*', missing, 'audio segments were not matched. A', float(missing)/len(pairs)*100, '% of the total')
        return video_start, video_end, audio_start, audio_end


    def print_transitions(self, transitions):
        for sign in transitions:
            logging.info('Sign starts at ' + str(sign.start))
            logging.info('Sign ends at ' + str(sign.end))
     
    def segment_video(self, block_id, audio_results_segmentation, video_files):
        results = []

        # OLD TRANSITIONS
        # detector = SignDetector()
        # transitions = detector.compute_transitions(self.audio_file, self.video_file)

        # suposing audio_results_segmentation = [{'start': 10, 'end':...}, {intervencions}, {}]

        # add to general pipleine after
        processor = CustomSignSegmentation(
            video_path = self.video_file,
            model_path = 'sign_language_segmentation/dist/model_E1s-1.pth'
        )
        
        video_segments = processor.process()


        #print(f'Segments var type: {type(segments)}')
        #print(f"Video segments {segments}")
        #print(f'There are {len(audio_results_segmentation)} audio results {audio_results_segmentation}')

        
        # matched_segments = self.__align_audio_and_video(audio_results_segmentation, segments)

        audio_segments = [(a['start'],a['end']) for a in audio_results_segmentation]
        pairs = self.__pair_audio_video_segments(audio_segments, video_segments, margin=6, threshold=0.0)

        for audio_result, pair in zip(audio_results_segmentation, pairs):
            #Remove old audio start and end 
            audio_result.pop('start',None)
            audio_result.pop('end',None)

            #Add new audio and sign segment
            audio_result['audio_start'] = pair['audio_start']
            audio_result['audio_end'] = pair['audio_end']
            audio_result['sign_start'] = pair['video_start']
            audio_result['sign_end'] = pair['video_end']

            results.append(audio_result)

        return results, video_segments
        
            
    def segment_audio(
        self,
        block_id: str,
        speaker: str,
        audio_files: dict,
    ) -> dict:
        results = []
        base_name = ".".join(os.path.basename(self.audio_file).split(".")[:-1])
        if base_name not in audio_files.keys():
            audio_file_name = "_".join(base_name.split("_")[:-2])
        else:
            audio_file_name = base_name

        if self.pipeline == "corts" or self.pipeline == "parlament":
            lang_model = Path(
                "data/models/Parla_Text_Corpus/5GRAM_Parla_Text_Corpus.bin"
            )
        else:
            lang_model = Path(
                "data/models/Parla_Text_Corpus/5GRAM_Parla_Text_Corpus.arpa"
            )

        for _, row in self.alignment.iterrows():
            start = row["start"]
            end = row["start"] + row["duration"]
            duration = row["duration"]
            text = row["text"].replace("<space>", " ")
            file_name, s3_out_segment_path = self.segment_cue(
                start, end, base_name, duration
            )
            paths_to_audio_files = [file_name]

            if self.pipeline == "corts" or self.pipeline == "parlament":
                if (
                    AudioSpeechRecognition.identify_language(self.lg_model, text)
                    == "__label__ca"
                ):
                    transcription = AudioSpeechRecognition.transcribe(
                        self.ca_model, paths_to_audio_files
                    )[0]
                    decoder = build_ctcdecoder(
                        self.ca_model.decoder.vocabulary,
                        kenlm_model_path=str(lang_model),  # either .arpa or .bin file
                        alpha=0.5,  # tuned on a val set
                        beta=1.0,  # tuned on a val set
                    )

                    transcription_lm = AudioSpeechRecognition.transcribe_with_model(
                        self.ca_model, paths_to_audio_files, decoder
                    )
                    language = "ca"
                else:
                    transcription = AudioSpeechRecognition.transcribe(
                        self.es_model, paths_to_audio_files
                    )[0]
                    decoder = build_ctcdecoder(
                        self.es_model.decoder.vocabulary,
                        kenlm_model_path=str(lang_model),  # either .arpa or .bin file
                        alpha=0.5,  # tuned on a val set
                        beta=1.0,  # tuned on a val set
                    )

                    transcription_lm = AudioSpeechRecognition.transcribe_with_model(
                        self.es_model, paths_to_audio_files, decoder
                    )
                    language = "es"
            else:
                if self._identify_language(self.lg_model, text) == "__label__ca":
                    decoder = build_ctcdecoder(
                        self.ca_model.decoder.vocabulary,
                        kenlm_model_path=str(lang_model),  # either .arpa or .bin file
                        alpha=0.5,  # tuned on a val set
                        beta=1.0,  # tuned on a val set
                    )
                    transcription = self._asr(self.ca_model, paths_to_audio_files)[0]
                    transcription_lm = self._asr_lm(
                        self.ca_model, decoder, paths_to_audio_files
                    )
                    language = "ca"
                else:
                    decoder = build_ctcdecoder(
                        self.es_model.decoder.vocabulary,
                        kenlm_model_path=str(lang_model),  # either .arpa or .bin file
                        alpha=0.5,  # tuned on a val set
                        beta=1.0,  # tuned on a val set
                    )
                    transcription = self._asr(self.es_model, paths_to_audio_files)[0]
                    transcription_lm = self._asr_lm(
                        self.es_model, decoder, paths_to_audio_files
                    )
                    language = "es"

            cer_score = cer(text, transcription)
            cer_score_lm = cer(text, transcription_lm)
            identifier = "_".join([base_name, str(start), str(end)])
            segment_path = s3_out_segment_path if self.remote_storage else file_name
            audio_extension = audio_files.get(audio_file_name, None)
            original_audio_path = self.__get_original_audio_path(
                audio_file_name, audio_extension
            )

            result = {
                "identifier": identifier,
                "block": block_id,
                "original_audio": original_audio_path,
                "speaker": speaker,
                "segment_path": segment_path,
                "start": start,
                "end": end,
                "language": language,
                "org_text": text,
                "pred_text": transcription,
                "pred_text_lm": transcription_lm,
                "cer_score": cer_score,
                "cer_score_lm": cer_score_lm,
            }

            results.append(result)

        return results

    def segment_cue(
        self, start: float, end: float, base_name: str, duration: float
    ) -> Tuple[str, str]:
        duration = str(duration)  # Start and End are 'Float' not 'int'.
        file_name = "_".join([base_name, str(start), str(end)]) + ".wav"
        file_path = os.path.join(self.out_path, file_name)
        s3_out_path = str(self.s3_out_path / file_name)

        try:
            if self.minio_client:
                self.minio_client.stat_object(str(self.s3_bucket), s3_out_path)
                msg = "%s already exists skipping" % file_path
                logging.info(msg)
                return file_path, str(self.s3_bucket / s3_out_path)
        except:
            pass

        args = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "panic",
            "-i",
            self.audio_file,
            "-ss",
            str(start),
            "-t",
            duration,
            "-ac",
            "1",
            "-ar",
            "16000",
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
        elif self.minio_client:
            logging.warning("Uploading file to MinIO...")
            self.minio_client.fput_object(str(self.s3_bucket), s3_out_path, file_path)
            return file_path, str(self.s3_bucket / s3_out_path)

        return file_path, None

    def _identify_language(self, model, text):
        predictions = model.predict(text, k=1)
        lang_id = predictions[0][0]
        return lang_id

    def _asr(self, model, paths_to_audio_files):
        transcription = model.transcribe(
            paths_to_audio_files=paths_to_audio_files, batch_size=1
        )
        return transcription

    def _asr_lm(self, model, decoder, paths_to_audio_files):
        logits = model.transcribe(
            paths_to_audio_files=paths_to_audio_files, logprobs=True
        )[0]
        transcription_with_lm = decoder.decode(logits)
        return transcription_with_lm
