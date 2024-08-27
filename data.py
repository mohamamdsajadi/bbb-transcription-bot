# data.py
from dataclasses import dataclass
from typing import Any, Optional, Union, Dict

import numpy as np

@dataclass
class AudioData:
    raw_audio_data: bytes
    audio_data: Optional[np.ndarray] = None
    transcribed_text: Optional[Dict[str, Union[str, Any]]] = None
    aligned_text: Optional[Dict[str, Union[str, Any]]] = None
