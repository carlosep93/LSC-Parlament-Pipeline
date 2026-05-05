import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from numpy.typing import NDArray

from dataclass_types import Transition
from typing import List


class SignTransitions:
    """
    Class that given a video of a person signing, calculates the time intervals
    where this person is moving their hands.
    """

    video_path: str
    sign_transitions: list[Transition]

    def __init__(self, video_path):
        self.video_path = video_path
        self.sign_transitions = []


    def calculate_motion_history(self, cap) -> NDArray[np.float64]:

        """
        Calculates the motion history of a video stream using the difference between frames.
        Filters are used to smooth the result and delte noise.
        """
        
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        motion_history = []

        # Filtre gaussià per reduir soroll
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)


        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            non_zero_count = np.count_nonzero(thresh)
            
            motion_history.append(non_zero_count)
            prev_gray = gray

        cap.release()


        # Aplicar el filtre gaussià a l'historial de moviment
        return gaussian_filter1d(motion_history, sigma=5)


    def transitions_detection(self, fps, smoothed_motion) -> list[tuple[float, float]]:
        """
        Function that detects using a thershold when the intervals (start, end) of the person that signs.
        """

        threshold = 10

        transitions = []
        for i in range(1, len(smoothed_motion)):
            if smoothed_motion[i-1] < threshold and smoothed_motion[i] >= threshold:
                transitions.append((i, 'start'))
            elif smoothed_motion[i-1] >= threshold and smoothed_motion[i] < threshold:
                transitions.append((i, 'stop'))

        return [(frame_number / fps, event) for frame_number, event in transitions]


    def filter_and_merge_transitions(self, transitions: List[tuple[tuple, float]], duration: float, short_th:int = 3, silence_th:float = 0.5) -> List[Transition]:
        
        intervals: list[tuple[float, float]] = list()

        if transitions[0][1] == 'stop':
            transitions.insert(0, (0, 'start'))

        if transitions[-1][1] == 'start':
            transitions.append((duration, 'end'))

        assert len(transitions) % 2 == 0, "incorrect sign intervals detection"

        for i in range(0, len(transitions) - 1, 2):
            intervals.append(Transition(transitions[i][0], transitions[i + 1][0]))

        # Step 1: Filter out transitions that are less than 4 seconds
        filtered_transitions = [t for t in intervals if t.duration() >= short_th]
        
        # Step 2: Merge adjacent transitions if the non-moving time between them is less than 1 second
        if not filtered_transitions:
            return []
        
        merged_transitions = [filtered_transitions[0]]
        
        for i in range(1, len(filtered_transitions)):
            previous = merged_transitions[-1]
            current = filtered_transitions[i]
            
            # Calculate the non-moving time between the previous and current transition
            non_moving_time = current.start - previous.end
            
            if non_moving_time < silence_th:
                # Merge the transitions
                merged_transitions[-1] = Transition(previous.start, current.end)
            else:
                # Add the current transition as is
                merged_transitions.append(current)
        
        return merged_transitions


    def calculate_transitions(self) -> list[Transition]:
        cap = cv2.VideoCapture(self.video_path)

        # Obtenir propietats del vídeo
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(frame_count, fps)
        duration = frame_count / fps

        smoothed_motion = self.calculate_motion_history(cap)
        transitions_seconds = self.transitions_detection(fps, smoothed_motion)

        self.sign_transitions = self.filter_and_merge_transitions(transitions_seconds, duration)

