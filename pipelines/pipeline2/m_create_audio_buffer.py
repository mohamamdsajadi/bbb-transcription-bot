from typing import List, Optional
from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule, Status
from stream_pipeline.module_classes import Module, ExecutionModule, ModuleOptions
import data
from extract_ogg import OggSFrame, calculate_frame_duration, get_header_frames

import logger

log = logger.get_logger()

class Create_Audio_Buffer(ExecutionModule):
    def __init__(self) -> None:
        super().__init__(ModuleOptions(
                                use_mutex=False,
                                timeout=5,
                            ),
                            name="Create_Audio_Buffer"
                        )
        self.audio_data_buffer: List[OggSFrame] = []
        self.last_n_seconds: int = 30
        self.min_n_seconds: int = 1
        self.current_audio_buffer_seconds: float = 0

    

        self.header_buffer: bytes = b''
        self.header_frames: Optional[List[OggSFrame]] = None

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not dp.data:
            raise Exception("No data found")
        if not dp.data.raw_audio_data:
            raise Exception("No audio data found")
        if not dp.data.sample_rate:
            raise Exception("No sample rate found")
            
        sample_rate = dp.data.sample_rate
            
        frame = OggSFrame(dp.data.raw_audio_data)

        if not self.header_frames:
            self.header_buffer += frame.raw_data
            id_header_frame, comment_header_frames = get_header_frames(self.header_buffer)

            if id_header_frame and comment_header_frames:
                self.header_frames = []
                self.header_frames.append(id_header_frame)
                self.header_frames.extend(comment_header_frames)
            else:
                dpm.message = "Could not find the header frames"
                dpm.status = Status.EXIT
                return

        

        last_frame: Optional[OggSFrame] = self.audio_data_buffer[-1] if len(self.audio_data_buffer) > 0 else None

        current_granule_position: int = frame.header['granule_position']
        previous_granule_position: int = last_frame.header['granule_position'] if last_frame else 0

        frame_duration: float = calculate_frame_duration(current_granule_position, previous_granule_position, sample_rate)
        previous_granule_position = current_granule_position


        self.audio_data_buffer.append(frame)
        self.current_audio_buffer_seconds += frame_duration

        # Every second, process the last n seconds of audio
        if frame_duration > 0.0 and self.current_audio_buffer_seconds >= self.min_n_seconds:
            if self.current_audio_buffer_seconds >= self.last_n_seconds:
                # pop audio last frame from buffer
                pop_frame = self.audio_data_buffer.pop(0)
                pop_frame_granule_position = pop_frame.header['granule_position']
                next_frame_granule_position = self.audio_data_buffer[0].header['granule_position'] if len(self.audio_data_buffer) > 0 else pop_frame_granule_position
                pop_frame_duration = calculate_frame_duration(next_frame_granule_position, pop_frame_granule_position, sample_rate)
                self.current_audio_buffer_seconds -= pop_frame_duration

            # Combine the audio buffer into a single audio package
            n_seconds_of_audio: bytes = self.header_buffer + b''.join([frame.raw_data for frame in self.audio_data_buffer])
            dp.data.raw_audio_data = n_seconds_of_audio
        else:
            dpm.status = Status.EXIT
            dpm.message = "Not enough audio data to create a package"