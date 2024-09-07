# m_create_audio_buffer.py
from typing import List, Optional

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule, Status
from stream_pipeline.module_classes import Module, ExecutionModule, ModuleOptions

from ogg import Ogg_OPUS_Audio, OggS_Page, calculate_page_duration
import data
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
        self.audio_data_buffer: List[OggS_Page] = []
        self.last_n_seconds: int = 10
        self.min_n_seconds: int = 1
        self.current_audio_buffer_seconds: float = 0

    

        self.header_buffer: bytes = b''
        self.header_pages: Optional[List[OggS_Page]] = None
        self.sample_rate: int = 0

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not dp.data:
            raise Exception("No data found")
        if not dp.data.raw_audio_data:
            raise Exception("No audio data found")
            
        page = OggS_Page(dp.data.raw_audio_data)

        if not self.header_pages:
            self.header_buffer += dp.data.raw_audio_data
            audio = Ogg_OPUS_Audio(self.header_buffer)
            print(f"Header Buffer: {len(self.header_buffer)}")
            # id_header_page, comment_header_pages = get_header_pages(self.header_buffer)
            id_header = audio.id_header
            comment_header = audio.comment_header

            if id_header and comment_header:
                self.sample_rate = id_header.input_sample_rate
                self.header_pages = []
                self.header_pages.append(OggS_Page(id_header.page.raw_data))
                self.header_pages.extend([OggS_Page(page.raw_data) for page in comment_header.pages])
            else:
                dpm.message = "Could not find the header pages"
                dpm.status = Status.EXIT
                return

        

        last_page: Optional[OggS_Page] = self.audio_data_buffer[-1] if len(self.audio_data_buffer) > 0 else None

        current_granule_position: int = page.granule_position
        previous_granule_position: int = last_page.granule_position if last_page else 0

        page_duration: float = calculate_page_duration(current_granule_position, previous_granule_position, self.sample_rate)
        previous_granule_position = current_granule_position


        self.audio_data_buffer.append(page)
        self.current_audio_buffer_seconds += page_duration

        # Every second, process the last n seconds of audio
        if page_duration > 0.0 and self.current_audio_buffer_seconds >= self.min_n_seconds:
            if self.current_audio_buffer_seconds >= self.last_n_seconds:
                # pop audio last page from buffer
                pop_page = self.audio_data_buffer.pop(0)
                pop_page_granule_position = pop_page.granule_position
                next_page_granule_position = self.audio_data_buffer[0].granule_position if len(self.audio_data_buffer) > 0 else pop_page_granule_position
                pop_page_duration = calculate_page_duration(next_page_granule_position, pop_page_granule_position, self.sample_rate)
                self.current_audio_buffer_seconds -= pop_page_duration

            # Combine the audio buffer into a single audio package
            n_seconds_of_audio: bytes = self.header_buffer + b''.join([page.raw_data for page in self.audio_data_buffer])
            dp.data.raw_audio_data = n_seconds_of_audio
        else:
            dpm.status = Status.EXIT
            dpm.message = "Not enough audio data to create a package"