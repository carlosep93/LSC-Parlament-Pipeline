import itertools
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from unicodedata import normalize as nrmlz

import yaml
from fuzzywuzzy import fuzz

from seq_aligner import finalize, needle

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)


class Alignment(object):
    def __init__(
        self, pipeline: str, data_directory: Path, sessions_directory_identifier: str
    ):
        """
        :param pipeline: The name of the data pipeline ["corts", "parlament"]
        :param data_directory: The relative path to the directory where data is stored
        :param sessions_directory_identifier: For the Parlament pipeline, this will be the session id. For the Corts pipeline,
                                              this will be the date
        """
        if not sessions_directory_identifier:
            raise ValueError("The session identifier cannot be None")

        self.sessions_directory_identifier = sessions_directory_identifier
        self.pipeline = pipeline
        self.data_directory = data_directory

        self.metadata = None
        self.metadata_speakers_list = []
        self.metadata_speakers_start_timestamps = []
        self.metadata_speakers_end_timestamps = []
        self.metadata_audios = []

        self.speakers = []

    def block_align(self):
        self.get_mesa()
        self.get_metadata_speakers()
        self.get_text_speakers()

        self.metadata_blocks = self.get_blocks(self.metadata_intervinents, self.mesa)
        self.text_blocks = self.get_blocks(self.text_intervinents, self.mesa)

        if len(self.metadata_blocks) != len(self.text_blocks):
            logging.warning(
                f"Alignment blocks are not the same size: {len(self.metadata_blocks)} vs {len(self.text_blocks)}"
            )

        self.match_speakers()
        if len(self.speaker_index) > 2:
            self.align_speakers()
        self.write_blocks()

    def get_mesa(self):
        first_meta_int = self.metadata_speakers_list[0]
        last_meta_int = self.metadata_speakers_list[-1]
        most_meta_int = Counter(self.metadata_speakers_list).most_common()[0][0]

        first_text_int = self.speakers[0]
        last_text_int = self.speakers[-1]
        most_text_int = Counter(self.speakers).most_common()[0][0]

        if first_meta_int == most_meta_int and last_meta_int == most_meta_int:
            self.metadata_mesa = first_meta_int
        else:
            logging.warning("Mesa person not found.")
            logging.info(
                f"First: {first_meta_int}, Last: {last_meta_int}, Most: {most_meta_int}"
            )

            if first_meta_int == most_meta_int:
                logging.info("Using first and the most")
                self.metadata_mesa = first_meta_int
            elif last_meta_int == first_meta_int:
                logging.info("Using first and the last")
                self.metadata_mesa = last_meta_int
            elif last_meta_int == most_meta_int:
                logging.info("Using most and the last")
                self.metadata_mesa = last_meta_int
            else:
                found = False
                for interv in [first_meta_int, last_meta_int, most_meta_int]:
                    if "parlament" in interv.lower():
                        logging.warning("Using the name with parlament in it")
                        found = True
                        self.metadata_mesa = interv
                        break
                if not found:
                    logging.warning("Using the most frequent")
                    self.metadata_mesa = most_meta_int

        if first_text_int == most_text_int and last_text_int == most_text_int:
            self.text_mesa = first_text_int
        else:
            logging.warning("Mesa person not found.")
            logging.info(
                f"First: {first_text_int}, Last: {last_text_int}, Most:  {most_text_int}"
            )

            if first_text_int == most_text_int:
                logging.info("Using the first and the most")
                self.text_mesa = first_text_int
            elif first_text_int == last_text_int:
                logging.info("Using the first and the last")
                self.text_mesa = last_text_int
            elif most_text_int == last_text_int:
                logging.info("Using the most and the last")
                self.text_mesa = last_text_int
            else:
                found = False
                for interv in [
                    first_text_int,
                    last_text_int,
                    most_text_int,
                ]:
                    if "parlament" in interv.lower():
                        logging.warning("Using the name with parlament in it")
                        found = True
                        self.text_mesa = interv
                        break
                if not found:
                    logging.warning("Using the most frequent")
                    self.text_mesa = most_text_int

        if self.pipeline == "corts":
            msg = "Mesa Diaris: %s |  Mesa Metadata: %s" % (
                self.text_mesa,
                self.metadata_mesa,
            )
        else:
            msg = "Mesa YAML: %s |  Mesa JSON: %s" % (
                self.text_mesa,
                self.metadata_mesa,
            )
        logging.info(msg)
        self.mesa = "mesa"

    def _process_metadata_speakers_list(self):
        """
        Extracts speakers and URLs from metadata,'
        processes speakers names and return a structured list of speakers
         for each intervantion
        """
        if not self.metadata_mesa:
            self.get_mesa()

        self.metadata_intervinents = []
        intervinents = []
        for intervinent in self.metadata_speakers_list:
            if self.metadata_mesa in intervinent:
                speaker = self.mesa  # put 'mesa' instead of real name of mesa
            else:
                speaker = intervinent.split("|")[0].strip().replace("\xa0", " ")
            intervinents.append(speaker)

        return intervinents

    def post_process_text_names(self, speakers):
        self.name_dict = {}
        self.text_speaker_normalizations = []
        for speaker in set(speakers):
            match = re.search("(.+)\((.+)\)", speaker)
            if match:
                title, name = match.groups()
                self.name_dict[self.normalize(speaker)] = name.strip()
                self.name_dict[self.normalize(title)] = name.strip()
        new = []
        for speaker in speakers:
            normalized_speaker = self.normalize(speaker)
            if self.name_dict.get(normalized_speaker):
                new.append(self.name_dict[normalized_speaker])
                self.text_speaker_normalizations.append(
                    (speaker, self.name_dict[normalized_speaker])
                )
            else:
                new.append(speaker)
        return new

    def normalize(self, input_str):
        return self.remove_accents(input_str.strip().lower())

    def write_blocks(self):
        blocks_dir = self.data_directory / "blocks"
        os.makedirs(blocks_dir, exist_ok=True)

        with open(blocks_dir / f"{self.sessions_directory_identifier}.blk", "w") as out:
            yaml.dump(
                [
                    compare
                    for compare in itertools.zip_longest(
                        self.metadata_blocks, self.text_blocks
                    )
                ],
                out,
            )

    def match_speakers(self):
        text_int_set = set(self.text_intervinents)
        metadata_int_set = set(self.metadata_intervinents)

        logging.info(
            f"Matching {len(text_int_set)} vs {len(metadata_int_set)} speakers"
        )
        text_int_set, metadata_int_set, matched_tvsm = self.match_loop(
            text_int_set, metadata_int_set, 90
        )
        if text_int_set:
            logging.info("There are unmatched PDF speakers")
            for t_int in list(text_int_set):
                for key, value in self.name_dict.items():
                    """
                    if the pdf speaker name is written in a different way (for
                    example a minister's title is missing a word) maybe it was
                    not picked up by the self.name_dict()

                    this loop assumes that the normalized version was already
                    matched and matched_tvsm should have it

                    name similarity might lead to false positives for 2 cases:
                    * el president (de generalitat) president(a) del parlament
                    * secretari mesa d'edat which there are multiple
                    """
                    if (
                        b"mesa" not in self.normalize(t_int)
                        and b"president" not in self.normalize(t_int)
                        and self.match_speaker(self.normalize(t_int), key, threshold=80)
                    ):
                        # not to end up with false positives of vicepresident
                        # of different types
                        logging.info(
                            f"{t_int} found in name similarity dict as {value}"
                        )
                        try:
                            matched_tvsm.append(
                                (t_int, self.find_in_tuple(matched_tvsm, value))
                            )
                            text_int_set.remove(t_int)
                        except Exception as e:
                            logging.error(
                                "The element in name_dict was"
                                " not matched with any db speaker"
                            )
                            logging.error(e)
                        break

        if text_int_set and metadata_int_set:
            text_int_set, metadata_int_set, new_matched_tvsm = self.match_loop(
                text_int_set, metadata_int_set, 80
            )
            if new_matched_tvsm:
                logging.info(
                    "Found more matches comparing the initial list of unmatched"
                )
            matched_tvsm += new_matched_tvsm
        if text_int_set or metadata_int_set:
            logging.info(f"Unmatched speakers: {text_int_set} vs {metadata_int_set}")
        else:
            logging.info("Success! All speakers matched.")
        self.get_speaker_index_tuples(matched_tvsm, text_int_set, metadata_int_set)
        self.matched_tvsm = matched_tvsm

    def match_loop(self, text_int_set, metadata_int_set, threshold):
        matched_tvsm = []
        if "mesa" in text_int_set or "mesa" in metadata_int_set:
            logging.info("Trying to find mesa vs mesa first")
            for t_int in list(text_int_set):
                for m_int in list(metadata_int_set):
                    if t_int == "mesa" and t_int == m_int:
                        logging.info("Mesa vs mesa found")
                        matched_tvsm.append((t_int, m_int))
                        metadata_int_set.remove(m_int)
                        text_int_set.remove(t_int)
        for t_int in list(text_int_set):
            if metadata_int_set:
                for m_int in list(metadata_int_set):
                    if self.match_speaker(
                        t_int.lower(), m_int.lower(), threshold=threshold
                    ):
                        matched_tvsm.append((t_int, m_int))
                        metadata_int_set.remove(m_int)
                        try:
                            text_int_set.remove(t_int)
                        except:
                            logging.warning(
                                f"PDF speaker {t_int} has two or more metadata counterparts {m_int}"
                            )
        return text_int_set, metadata_int_set, matched_tvsm

    def match_speaker(self, t_int, m_int, threshold=90):
        return fuzz.token_set_ratio(t_int, m_int) > threshold

    def get_speaker_index_tuples(self, tvsm, t_set, m_set):
        """Converts the matched and unmatched speaker lists into one list of
        speaker(s) vs index tuples.
        Does not yet take into account the repetitions (only one-to-one)
        """
        self.speaker_index = []
        index = 0
        for tm in tvsm:
            self.speaker_index.append((tm[0], tm[1], index))
            index += 1
        for t in t_set:
            self.speaker_index.append((t, "", index))
            index += 1
        for m in m_set:
            self.speaker_index.append(("", m, index))
            index += 1

    def align_speakers(self):
        """Uses the built blocks to align the speakers
        Otherwise the repetitions in one block hinders the discovery
        of "good" alignments
        """
        self.normalized_many2many_dict = {}
        text_block_int = [e[2] for e in self.text_blocks]
        meta_block_int = [e[2] for e in self.metadata_blocks]
        text_int_seq = self.get_sequence(text_block_int, 0)
        meta_int_seq = self.get_sequence(meta_block_int, 1)

        text_int_seq = self.replace_many2many(text_int_seq)
        meta_int_seq = self.replace_many2many(meta_int_seq)
        text_int_aligned, meta_int_aligned = needle(text_int_seq, meta_int_seq)

        finalize(text_int_aligned, meta_int_aligned)
        self.text_blocks = self.sequence2name(text_int_aligned, self.text_blocks, 0)
        self.metadata_blocks = self.sequence2name(
            meta_int_aligned, self.metadata_blocks, 1
        )

    def get_sequence(self, name_seq, name_index):
        seq = []
        for name in name_seq:
            index = [ref[2] for ref in self.speaker_index if ref[name_index] == name]
            for i in index[1:]:
                self.normalized_many2many_dict[i] = index[0]
            # if more than one index is found, the first index will be assigned
            # they are stored in the normalized_many2many_dict
            # the indicies will be replaced accordingly after both sequences
            # are populated
            seq.append(index[0])
        return seq

    def replace_many2many(self, seq):
        new_seq = []
        for s in seq:
            replace = self.normalized_many2many_dict.get(s)
            if replace:
                new_seq.append(replace)
            else:
                new_seq.append(s)
        return new_seq

    def sequence2name(self, seq, blocks, index):
        new_blocks = []
        for s in seq[::-1]:
            if s != "--":
                name = self.speaker_index[int(s)][index]
                if not name == blocks[-1][2]:
                    # TODO unless we invert replace_many2many operation
                    # there is nothing else to do. We just have to trust
                    # the index order
                    logging.warning(
                        f"Name is not as expected. Could be another equivalent name: {s}, {name}, {blocks[-1][2]}"
                    )
                new_blocks.append(blocks[-1])
                blocks.pop()
            else:
                new_blocks.append((None, None, None))
        return new_blocks[::-1]

    def output_media_vs_text(self):
        sessions_directory = self.data_directory / "sessions"
        session_id_directory = (
            sessions_directory / f"{self.sessions_directory_identifier}"
        )
        os.makedirs(session_id_directory, exist_ok=True)

        # output aligned speaker list
        speaker_file_path = session_id_directory / "aligned_speakers.ls"
        speakers_list = [(self.text_mesa, self.metadata_mesa.split("|")[0])] + [
            (sp[0], sp[1]) for sp in self.speaker_index[1:]
        ]

        with open(speaker_file_path, "w") as sp_out:
            yaml.dump(speakers_list, sp_out)
        # save only the speakers with a match
        self.text_speakers_set = set(
            [speakers[0] for speakers in self.matched_tvsm]
            + [speakers[0] for speakers in self.text_speaker_normalizations]
            + [speakers[1] for speakers in self.text_speaker_normalizations]
        )
        self.meta_speakers_set = set([sp[1] for sp in self.matched_tvsm])
        best_blocks = self.get_best_aligned_pairs()

        # TODO: Use a more descriptive name for this file
        out_file = (
            session_id_directory / f"output_{self.sessions_directory_identifier}.json"
        )
        count = 0
        with open(out_file, "w", encoding="utf-8") as f:
            aligned_text_data = {}
            for best_text, best_meta in best_blocks:
                if self.pipeline == "parlament":
                    (
                        aligned_text,
                        speaker_meta,
                        start,
                        end,
                        url,
                        audio,
                    ) = self.blocks_contents(best_text, best_meta)
                    values = {
                        "text": aligned_text,
                        "speaker_meta": speaker_meta,
                        "start": str(start),
                        "end": str(end),
                        "metadata_url": url,
                    }
                    if audio.startswith("parlament/audios/"):
                        values["path_audio"] = audio
                    else:
                        values["path_segment_audio"] = audio
                elif self.pipeline == "corts":
                    (
                        aligned_text,
                        speaker_meta,
                        start,
                        end,
                        record,
                    ) = self.blocks_contents(best_text, best_meta)
                    values = {
                        "text": aligned_text,
                        "speaker_meta": speaker_meta,
                        "start": str(start),
                        "end": str(end),
                        "record": str(record),
                    }
                aligned_text_data[count] = values
                count += 1
            json.dump(aligned_text_data, f, indent=4, ensure_ascii=False)

            return aligned_text_data

    def get_best_aligned_pairs(self):
        """Checks the aligned blocks for good alignments"""
        good_blocks = []
        alignment = list(itertools.zip_longest(self.text_blocks, self.metadata_blocks))
        for i, block_row in enumerate(
            itertools.zip_longest(self.text_blocks, self.metadata_blocks)
        ):
            # TODO if 100% match use also the beginning and the end
            if i == 0 or i == len(alignment) - 1:
                continue
            block_is_good = True
            # If the row is surrounded by aligned rows of blocks it is good
            for row in alignment[i - 1 : i + 2]:
                if block_is_good:
                    for block in row:
                        if not block[2]:
                            block_is_good = False
                            break
            if block_is_good:
                if block_row[0][1] and block_row[1][1]:
                    if self.block_in_matched(block_row):
                        good_blocks.append(block_row)
        return good_blocks

    def block_in_matched(self, block_row):
        text_speaker = block_row[0][2]
        meta_speaker = block_row[1][2]
        if (
            text_speaker not in self.text_speakers_set
            or meta_speaker not in self.meta_speakers_set
        ):
            logging.warning(
                f"Bad alignment? One of them is not in the list of matched speakers: {str(block_row)}"
            )
            # It is possible that sequence to sequence alignment
            # serendipitously aligned a correct pair
            # checking if it is the case
            text_sp_norm = re.sub("\<.+?\>", "", text_speaker).lower()
            meta_sp_norm = meta_speaker.lower()
            if self.match_speaker(text_sp_norm, meta_sp_norm, threshold=60):
                msg = "They are a good match"
                logging.info(msg)
            else:
                return False
        return True

    @staticmethod
    def post_process_db(speakers):
        """
        :param speakers: 2D list of speakers
        Processes the intervinent list from the db elements, in order to extract a
        single speaker.
        WARNING: In some cases we might still end up with multiple speakers
        """
        processed_speaker_list = []
        for speaker_list in speakers:
            if len(speaker_list) == 1:
                processed_speaker_list.append(speaker_list[0])
            else:
                logging.info(f"Postprocessing {speaker_list}")
                index = -1
                if "mesa" in speaker_list:
                    index = speaker_list.index("mesa")
                    speaker_list.pop(index)
                else:
                    logging.warning('Determined "mesa" not in list')
                if len(speaker_list) > 1:
                    logging.info(
                        f"After post-process there are still multiple intervinents: {speaker_list}\nAdding only the first one"
                    )
                processed_speaker_list.append(speaker_list[0])
        return processed_speaker_list

    @staticmethod
    def remove_accents(input_str):
        nfkd_form = nrmlz("NFKD", input_str)
        only_ascii = nfkd_form.encode("ASCII", "ignore")
        return only_ascii

    @staticmethod
    def get_blocks(metadata_intervinents, pause):
        """slists the metadata_intervinents list into  blocks of consecutive items
        and returns a tuple list of start/end indices + unique/mesa(pause) items (each block)
        Each block contains a unique item and includes consecuitive items until the
        next unique item  or mesa(pause) is found"""
        uniques = []  # unique items exlcuding pauses
        for p in metadata_intervinents:
            if not p == pause:
                if not uniques:
                    uniques.append(p)
                else:
                    if p != uniques[-1]:
                        uniques.append(p)
        # searches for beginning and end indices of the blocks
        search_beginning = False
        blocks = []
        end = 0
        for u in uniques:
            search_beginning = True  # iterates over items and indices in uniques
            for i, p in enumerate(metadata_intervinents):
                if i >= end:
                    if search_beginning:
                        if p == pause or p == u:
                            start = i
                            search_beginning = False
                    else:
                        if p != u and p != pause:
                            end = i - 1
                            break
            search_beginning = True
            if end < start:
                # in the case the end of the list is reached without a match
                end = start
            blocks.append(
                (start, end, u)
            )  # appends start/end indices and uniques items
        return blocks

    @staticmethod
    def find_in_tuple(tpl, val):
        """
        in a tuple searches val within in the first elements,
        returns the corresponding second element
        """
        for t in tpl:
            if t[0] == val:
                return t[1]
        msg = "ERROR: tuple does not have %s in its first elements" % (str(val))
        raise ValueError(msg)
