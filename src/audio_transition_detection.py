import collections
import contextlib
import sys
import wave
import webrtcvad
from dataclass_types import Transition, Frame

class AudioTransitions:
    """Class that given a audio file, returns the intervals when a person is speaking."""

    path: str
    transitions: list[Transition]    

    def __init__(self, path):
        self.path = path
        self.transitions = ()

    def read_wave(self):
        """Reads a .wav file.

        Takes the path, and returns (PCM audio data, sample rate).
        """
        path = self.path
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000, 48000)
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate

    def write_wave(self, audio, sample_rate):
        """Writes a .wav file.

        Takes path, PCM audio data, and sample rate.
        """

        with contextlib.closing(wave.open(self.path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)

    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        """Generates audio frames from PCM audio data.

        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.

        Yields Frames of the requested duration.
        """
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self, sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
        """Filters out non-voiced audio frames and returns their start and end times.

        Given a webrtcvad.Vad and a source of audio frames, yields only
        the start and end times of the non-voiced audio.

        Uses a padded, sliding window algorithm over the audio frames.
        When more than 90% of the frames in the window are voiced (as
        reported by the VAD), the collector triggers and begins yielding
        audio frames. Then the collector waits until 90% of the frames in
        the window are unvoiced to detrigger.

        The window is padded at the front and back to provide a small
        amount of silence or the beginnings/endings of speech around the
        voiced frames.

        Arguments:

        sample_rate - The audio sample rate, in Hz.
        frame_duration_ms - The frame duration in milliseconds.
        padding_duration_ms - The amount to pad the window, in milliseconds.
        vad - An instance of webrtcvad.Vad.
        frames - a source of audio frames (sequence or generator).
        time_windows - a list of tuples (start, end) indicating the time windows to process.

        Returns: A list of tuples containing (start_time, end_time) for voiced audio within the time windows.
        """
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        voiced_segments = []

        for frame in frames:
            
            is_speech = vad.is_speech(frame.bytes, sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])

                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    voiced_start = ring_buffer[0][0].timestamp
                    ring_buffer.clear()
            else:
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])

                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    voiced_end = frame.timestamp + frame.duration
                    triggered = False
                    voiced_segments.append(Transition(voiced_start, voiced_end))
                    ring_buffer.clear()

        if triggered:
            voiced_end = frame.timestamp + frame.duration
            voiced_segments.append(Transition(voiced_start, voiced_end))
            triggered = False

        return voiced_segments



    def calculate_transitions (self) -> list[Transition]:
        audio, sample_rate = self.read_wave()
        vad = webrtcvad.Vad(3)
        frames = self.frame_generator(30, audio, sample_rate)
        frames = list(frames)
        self.transitions = self.vad_collector(sample_rate, 30, 300, vad, frames)


