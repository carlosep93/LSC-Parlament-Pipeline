from dataclass_types import Transition 
from transitions_validation import Validator
from audio_transition_detection import AudioTransitions
from sign_transition_detection import SignTransitions
from merge_audio_sign_intervals import find_closest_intervals
import json

import os
import sys


class SignDetector():
    def __init__(self, verbose=False):
        self.validator = Validator()
        

    def save_transitions_to_file(self, transitions: list[Transition], action: str) -> None:
        folder_path = 'actions'
        file_path = os.path.join(folder_path, action + '.txt')
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        with open(file_path, 'w') as file:
            for trans in transitions:
                file.write(f'Starts {action} at {trans.start_minute} and ends at {trans.end_minute} minutes\n')
            file.write("\n")

    def transitions_to_dict(self,sign_audio_transitions_merged: list[tuple[Transition, Transition]]) ->  dict:
        signed_segments = {}
        id = 0 
        for sign, audio in sign_audio_transitions_merged:
            id += 1
            if self.verbose:
                print(f'Starts speaking at {audio.start_minute} and signing at {sign.start_minute} ')
                print(f'Ends speaking at {audio.end_minute} and signing at {sign.end_minute} ')
            
            signed_segments[id] = { 'sign_start':sign.start,
                                    'sign_end': sign.end,
                                    'audio_start':audio.start,
                                    'audio_end': audio.end}

            return signed_segments

    def compute_transitions(self, audio_path: str, video_path: str) -> None:
        #sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        # video example
        sign_trans = SignTransitions(video_path)

        # calculate sign transitions & check they are correct
        sign_trans.calculate_transitions()
        sign_transitions = sign_trans.sign_transitions
        
        '''
        self.validator.set_sign(sign_transitions)
        audio_trans = AudioTransitions(audio_path)

        # calculate audio transitions & check they are correct
        audio_trans.calculate_transitions()
        audio_transitions = audio_trans.transitions

        self.validator.set_audio(audio_transitions)
        self.validator.audio_validator()
        
        # merging both transitions
        sign_audio_transitions_merged: list[tuple[Transition, Transition]] = find_closest_intervals(sign_transitions, audio_transitions)
        '''

        return sign_transitions
    
