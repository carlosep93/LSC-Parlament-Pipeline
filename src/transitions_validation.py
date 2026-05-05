from dataclass_types import Transition


class Validator:
    """
    Class that is used to check that the transitions detected are consistent and coherent.
    """

    sign: list[Transition] 
    audio: list[Transition]

    def __init__(self):
        self.sign = []
        self.audio = []

    def set_sign(self, sign: list[Transition]) -> None:
        self.sign = sign

    def set_audio(self, audio: list[Transition]) -> None:
        self.audio = audio

    def sign_validator(self):
        """Checks that sign transitions are correctly computed."""

        last_start = self.sign[0].start
        last_end = self.sign[0].end

        for i, sign_tr in enumerate(self.sign):
            start, end = sign_tr.start, sign_tr.end

            assert (end - start) >= 4, "error while removing short sign intervals ({0} and {1}) of duration {2}".format(sign_tr.start_minute, sign_tr.end_minute, end-start)
            if i == 0: continue

            assert start > last_start, "start timings not stricly ascending"
            assert end > last_end, "end timings not stricly ascending"
            assert start - last_end >= 1, "error while removing short silence intervals"

            last_end = end 
            last_start = start

    def audio_validator(self):
        """Checks that audio transitions are correctly computed."""
        
        last_start = self.audio[0].start
        last_end = self.audio[0].end

        for i, audio_tr in enumerate(self.audio):
            start, end = audio_tr.start, audio_tr.end

            if i == 0: continue

            assert start > last_start, "start timings not stricly ascending"
            assert end > last_end, "end timings not stricly ascending"
            # assert start - last_end >= 1, "error while removing short silence intervals"

            last_end = end 
            last_start = start
        
    def audio_sign_validator (self) -> None:
        """Checks if the signing and audio transitions are correctly computed and coherent"""
        
        assert len(self.sign) == len(self.audio), "transitions are not the same length"
        
        for s, a in zip(self.sign, self.audio):
            if not (s.start > a.start and s.end > a.end):
                print(s.start_minute, a.start_minute, s.end_minute, a.end_minute)
            assert (s.start > a.start and s.end > a.end), "wrong audio sign syncronization"  

        return True
