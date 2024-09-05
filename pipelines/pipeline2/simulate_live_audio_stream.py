import time
from typing import Callable, List, Optional
from extract_ogg import OggSFrame, calculate_frame_duration, extract_id_header_frame, get_sample_rate, split_ogg_data_into_frames


def simulate_live_audio_stream(file_path: str, callback: Callable[[bytes], None]) -> None:
    with open(file_path, 'rb') as file:
        ogg_bytes: bytes = file.read()

    frames: List[OggSFrame] = split_ogg_data_into_frames(ogg_bytes)
    id_header_frame = extract_id_header_frame(frames)
    if id_header_frame is None:
        raise ValueError("No ID header frame found")
    sample_rate = get_sample_rate(id_header_frame)

    previous_granule_position: Optional[int] = None
    for frame_index, frame in enumerate(frames):
        current_granule_position: int = frame.header['granule_position']
        frame_duration: float = calculate_frame_duration(current_granule_position, previous_granule_position, sample_rate)
        previous_granule_position = current_granule_position

        # Sleep to simulate real-time audio playback
        time.sleep(frame_duration)
        
        callback(frame.raw_data)