from dataclasses import dataclass

@dataclass 
class Transition:
    """Represents a Transition (start speaking / signing) -> (stop speaking / signing)"""

    start: float
    end: float
    start_minute: str
    end_minute: str

    @staticmethod
    def seconds_to_hhmmss(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{secs:02}"
    
    def duration(self):
        return self.end - self.start
    
    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end
        self.start_minute = self.seconds_to_hhmmss(start)
        self.end_minute = self.seconds_to_hhmmss(end)


@dataclass
class Frame:
    """Represents a Frame of audio data"""
    bytes: bytes
    timestamp: float
    duration: float

