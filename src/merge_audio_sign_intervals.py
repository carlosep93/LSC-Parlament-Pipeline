from typing import Optional, List

from dataclass_types import Transition


def binary_search_closest_start(transitions: List[Transition], target: int) -> Optional[Transition]:
    left, right = 0, len(transitions) - 1
    closest = None
    
    while left <= right:
        mid = (left + right) // 2
        if transitions[mid].start < target:
            closest = transitions[mid]
            left = mid + 1
        else:
            right = mid - 1
    
    return closest

def binary_search_closest_end(transitions: List[Transition], target: int) -> Optional[Transition]:
    left, right = 0, len(transitions) - 1
    closest = None
    
    while left <= right:
        mid = (left + right) // 2
        if transitions[mid].end < target:
            closest = transitions[mid]
            left = mid + 1
        else:
            right = mid - 1
    
    return closest

def find_closest_intervals(sign_trans: List[Transition], audio_trans: List[Transition]) -> List[tuple[Transition, Optional[Transition]]]:
    """Returns """

    results = []
    
    for sign_trans in sign_trans:
        closest_start = binary_search_closest_start(audio_trans, sign_trans.start)
        closest_end = binary_search_closest_end(audio_trans, sign_trans.end)
        if closest_start.end < sign_trans.start: continue
        if closest_end.start > sign_trans.end: continue
        results.append((sign_trans, Transition(closest_start.start, closest_end.end)))
    
    return results



