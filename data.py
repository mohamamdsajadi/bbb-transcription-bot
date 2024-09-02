# data.py
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union, Dict

import numpy as np

@dataclass
class TextSegment:
    text: str
    start: float
    end: float

@dataclass
class AudioData:
    raw_audio_data: bytes                                    # Current audio chunk from stream
    sample_rate: int                                         # Sample rate of the audio data
    audio_buffer: Optional[bytes] = None                     # Buffer of n seconds of raw audio data
    audio_data: Optional[np.ndarray] = None                  # Audio data converted to mono waveform
    audio_data_sample_rate: Optional[int] = None             # Sample rate of the audio data after conversion
    vad_result: Optional[List[Dict[str, float]]] = None      # Voice activity detection result
    language: Optional[Tuple[str, float]] = None             # Detected language of the audio data (language code, probability)
    transcribed_segments: Optional[List[TextSegment]] = None # Transcribed segments as text with timestamps
    transcribed_text: Optional[str] = None                   # Full transcribed text
    transcribed_words: Optional[List[str]] = None            # List of transcribed words
    confirmed_words: Optional[List[str]] = None              # List of confirmed words
    unconfirmed_words: Optional[List[str]] = None            # List of unconfirmed words