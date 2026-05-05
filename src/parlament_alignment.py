import json
import os
import re
from pathlib import Path

import yaml

from alignment import Alignment


class ParlamentAlignment(Alignment):
    def __init__(self, pipeline: str, data_directory: Path, session_id: str):
        super().__init__(pipeline, data_directory, session_id)

        self.yaml_directory = self.data_directory / "yamls"
        self.json_directory = self.data_directory / "jsons"
        self.audio_directory = self.data_directory / "audios" 

        self.yaml_files = [
            file for file in self.yaml_directory.glob(f"{session_id}_*_speaker.yaml")
        ]
        if len(self.yaml_files) == 0:
            raise IOError(f"no yaml files with {session_id}")
        self.yaml_files.sort()

        self.json_files = [
            file for file in self.json_directory.glob(f"{session_id}_*.json")
        ]
        if len(self.json_files) == 0:
            raise IOError(f"no json files with {session_id}")
        self.json_files.sort()
        
        self.audio_files = [
            file for file in self.audio_directory.glob(f"{session_id}_*.wav")
        ]
        if len(self.audio_files) == 0:
            raise IOError(f"no audio files with {session_id}")
        self.audio_files

        self.get_metadata()
        self.get_text()

    def post_process_text_intervention(self, interventions):
        tag_pattern = re.compile(r"<[^>]+>")  # Remove HTML tags
        paren_pattern = re.compile(r"\([^)]*\)")  # Remove content within parentheses

        processed_interventions = []

        for speaker, talk in interventions:
            talk = tag_pattern.sub("", talk)  # Remove HTML tags
            talk = paren_pattern.sub("", talk)  # Remove content within parentheses

            processed_interventions.append((speaker, talk))
        return processed_interventions

    def get_metadata(self):
        self.metadata_urls = []
        self.metadata = []

        for json_file in self.json_files:
            if os.path.isfile(json_file):
                sessions = json.load(open(json_file))["data"]["sessions"]
                for s in sessions:
                    self.metadata.append((json_file,s))

        for json_file, intervention in self.metadata:
            #audio_path = str(json_file).replace(".json", ".wav").replace("/jsons/", "/audios/")
            audio_file = intervention['path_video'].split('/')[-1].replace('.mp4', '.wav') 
            audio_path = str(self.audio_directory) +  "/" +  audio_file
            if not os.path.isfile(audio_path):
                continue
            '''
            audio_path = None
            if "path_audio" in intervention:
                audio_path = intervention.get("path_audio")
            elif "path_segment_audio" in intervention:
                audio_path = intervention.get("path_segment_audio")
            # if audio path is not found, skip to the next intervention
            if audio_path is None:
                continue
            '''

            self.metadata_speakers_list += [intervention["intervinent_nom"]]
            self.metadata_speakers_start_timestamps += [
                intervention["intervencio_inici_relatiu"]
            ]
            self.metadata_speakers_end_timestamps += [
                intervention["intervencio_final_relatiu"]
            ]

            if "video_qaudio_mp4_url" in intervention:
                self.metadata_urls.append(intervention["video_qaudio_mp4_url"])
            elif "video_qminima_mp4_url" in intervention:
                self.metadata_urls.append(intervention["video_qminima_mp4_url"])
            elif "video_intervencio_url_download" in intervention:
                self.metadata_urls.append(
                    intervention["video_intervencio_url_download"]
                )

            self.metadata_audios += [audio_path]

        url_string = "".join(self.metadata_urls)
        delimiter = "https://"
        self.metadata_urls = [
            delimiter + item for item in url_string.split(delimiter) if item
        ]

    def get_text(self):
        self.text = []
        self.speakers_intervention = []
        for yaml_file in self.yaml_files:
            self.text += yaml.safe_load(open(yaml_file))
            if not self.text:
                msg = "%s yaml text empty" % self.sessions_directory_identifier
                raise ValueError(msg)

        for intervention in self.text:
            speaker, speaker_intervention = (
                list(intervention.keys())[0],
                list(intervention.values())[0],
            )
            self.speakers.append(speaker)
            self.speakers.append(speaker_intervention)

    def get_metadata_speakers(self):
        structured_metadata_intervinents = self._process_metadata_speakers_list()
        self.metadata_intervinents = self.post_process_db(
            [structured_metadata_intervinents]
        )
        self.metadata_intervinents = structured_metadata_intervinents

        return self.metadata_intervinents

    def get_text_speakers(self):
        """extracts speakers + their talks from texts.
         Structures the speakers and also stores the speaker, text tuple
        for later, for the output
        """
        self.text_intervinents = []  # list of speakers but with 'mesa'
        self.text_interventions = (
            []
        )  # list of tuple with mesa's real name and other speakers + intervention
        for d in self.text:
            speaker = list(d.keys())[0]  # extracts speaker's name
            talk = d[speaker]  # extracts the talk
            # check if talk is non-empty, 'ORDRE' is not speaker name and it doesn't have a capital letter
            if talk:
                if speaker == self.text_mesa:  # check if speaker is mesa
                    speaker = self.mesa  # replace name with 'mesa'
                self.text_intervinents.append(
                    speaker
                )  # append speaker as 'mesa' in the intervnients list
                self.text_interventions.append(
                    (speaker, talk)
                )  # append 'mesa' + other speakers ID and its talk in a tuple

        self.text_intervinents = self.post_process_text_names(self.text_intervinents)
        self.text_interventions = self.post_process_text_intervention(
            self.text_interventions
        )
        return self.text_intervinents

    def blocks_contents(self, text_block, metadata_block):
        text_start_i, text_end_i = text_block[:2]
        json_start_i, json_end_i = metadata_block[:2]

        return (
            self.text_interventions[text_start_i:text_end_i],
            self.metadata_speakers_list[json_start_i + 1],
            self.metadata_speakers_start_timestamps[json_start_i + 1],
            self.metadata_speakers_end_timestamps[json_end_i + 1],
            self.metadata_urls[json_start_i + 1],
            self.metadata_audios[json_start_i + 1],
        )
