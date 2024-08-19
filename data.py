

from dataclasses import dataclass


@dataclass
class AudioData:
    raw_audio_data: bytes
    text: str = ""