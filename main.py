import io
import json
import tempfile
import time
from typing import List, Optional
import whisper # type: ignore
import torch
from pydub import AudioSegment # type: ignore
from stream_pipeline.grpc_server import GrpcServer
from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ExecutionModule, ModuleOptions
from stream_pipeline.pipeline import Pipeline, ControllerMode, PipelinePhase, PipelineController

import data
from extract_ogg import get_header_frames, split_ogg_data_into_frames, OggSFrame
import logger

log = logger.setup_logging()

def calculate_frame_duration(current_granule_position, previous_granule_position, sample_rate=48000):
    if previous_granule_position is None:
        return 0.0  # Default value for the first frame
    samples = current_granule_position - previous_granule_position
    duration = samples / sample_rate
    return duration

class CreateNsAudioPackage(ExecutionModule):
    def __init__(self) -> None:
        super().__init__(ModuleOptions(
                                use_mutex=False,
                                timeout=5,
                            ),
                            name="Create10sAudioPackage"
                        )
        self.audio_data_buffer: List[OggSFrame] = []
        self.sample_rate: int = 48000
        self.last_n_seconds: int = 10
        self.current_audio_buffer_seconds: float = 0

    

        self.header_buffer: bytes = b''
        self.header_frames: Optional[List[OggSFrame]] = None

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if dp.data:
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
                    return

            

            last_frame: Optional[OggSFrame] = self.audio_data_buffer[-1] if len(self.audio_data_buffer) > 0 else None

            current_granule_position: int = frame.header['granule_position']
            previous_granule_position: int = last_frame.header['granule_position'] if last_frame else 0

            frame_duration: float = calculate_frame_duration(current_granule_position, previous_granule_position, self.sample_rate)
            previous_granule_position = current_granule_position


            self.audio_data_buffer.append(frame)
            self.current_audio_buffer_seconds += frame_duration

            # Every second, process the last n seconds of audio
            if frame_duration > 0.0:
                if self.current_audio_buffer_seconds >= self.last_n_seconds:
                    # pop audio last frame from buffer
                    pop_frame = self.audio_data_buffer.pop(0)
                    pop_frame_granule_position = pop_frame.header['granule_position']
                    next_frame_granule_position = self.audio_data_buffer[0].header['granule_position'] if len(self.audio_data_buffer) > 0 else pop_frame_granule_position
                    pop_frame_duration = calculate_frame_duration(next_frame_granule_position, pop_frame_granule_position, self.sample_rate)
                    self.current_audio_buffer_seconds -= pop_frame_duration

                # Combine the audio buffer into a single audio package
                n_seconds_of_audio: bytes = self.header_buffer + b''.join([frame.raw_data for frame in self.audio_data_buffer])
                dp.data.raw_audio_data = n_seconds_of_audio



class Whisper(Module):
    def __init__(self) -> None:
        super().__init__(ModuleOptions(
                                use_mutex=True,
                                timeout=5,
                            ),
                            name="Whisper"
                        )
        self.ram_disk_path = "/mnt/ramdisk" # string: Path to the ramdisk
        self.task = "translate"             # string: transcribe, translate (transcribe or translate it to english)
        self.model = "large"                # string: tiny, base, small, medium, large (Whisper model to use)
        self.models_path = ".models"        # string: Path to the model
        self.english_only = False           # boolean: Only translate to english

        if self.model != "large" and self.english_only:
            self.model = self.model + ".en"

        self._whisper_model: Optional[whisper.Whisper] = None
        
    def init_module(self) -> None:
        log.info(f"Loading model '{self.model}'...")
        self._whisper_model = whisper.load_model(self.model, download_root=self.models_path)
        log.info("Model loaded")
    
    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not self._whisper_model:
            raise Exception("Whisper model not loaded")
        if dp.data:
            log.info(f"Processing {len(dp.data.raw_audio_data)} bytes of audio data")
            if dp.data:
                with tempfile.NamedTemporaryFile(prefix='tmp_audio_', suffix='.wav', dir=self.ram_disk_path, delete=True) as temp_file:
                    # Convert opus to wav
                    opus_data = io.BytesIO(dp.data.raw_audio_data)
                    opus_audio = AudioSegment.from_file(opus_data, format="ogg", frame_rate=48000, channels=2, sample_width=2)
                    opus_audio.export(temp_file.name, format="wav")
                    
                    # Transcribe audio data
                    result = self._whisper_model.transcribe(temp_file.name, fp16=torch.cuda.is_available(), task=self.task)
                    text = result['text'].strip()
                    log.info(f"Transcribed text: {text}")

controllers = [
    PipelineController(
        mode=ControllerMode.NOT_PARALLEL,
        max_workers=1,
        name="CreateNsAudioPackage",
        phases=[
            PipelinePhase(
                name="CreateNsAudioPackagePhase",
                modules=[
                    CreateNsAudioPackage()
                ]
            )
        ]
    ),

    PipelineController(
        mode=ControllerMode.FIRST_WINS,
        max_workers=10,
        name="MainProcessingController",
        phases=[
            PipelinePhase(
                name="WhisperPhase",
                modules=[
                    Whisper()
                ]
            )
        ]
    )
]

pipeline = Pipeline[data.AudioData](controllers, name="WhisperPipeline")

def callback(dp: DataPackage[data.AudioData]) -> None:
    log.info("Callback", extra={"data_package": dp})
    
def exit_callback(dp: DataPackage[data.AudioData]) -> None:
    log.info("Exit", extra={"data_package": dp})

def overflow_callback(dp: DataPackage[data.AudioData]) -> None:
    log.info("Overflow", extra={"data_package": dp})

def outdated_callback(dp: DataPackage[data.AudioData]) -> None:
    log.info("Outdated", extra={"data_package": dp})

def error_callback(dp: DataPackage[data.AudioData]) -> None:
    log.error("Pipeline error", extra={"data_package": dp})

instance = pipeline.register_instance()

def simulate_live_audio_stream(file_path: str, sample_rate: int = 48000) -> None:
    with open(file_path, 'rb') as file:
        ogg_bytes: bytes = file.read()

    frames: List[OggSFrame] = split_ogg_data_into_frames(ogg_bytes)  # Assuming Frame is the type for frames
    previous_granule_position: Optional[int] = None

    for frame_index, frame in enumerate(frames):
        current_granule_position: int = frame.header['granule_position']
        frame_duration: float = calculate_frame_duration(current_granule_position, previous_granule_position, sample_rate)
        previous_granule_position = current_granule_position

        # Sleep to simulate real-time audio playback
        time.sleep(frame_duration)

        audio_data: data.AudioData = data.AudioData(raw_audio_data=frame.raw_data)

        pipeline.execute(audio_data, instance, callback=callback, exit_callback=exit_callback, overflow_callback=overflow_callback, outdated_callback=outdated_callback, error_callback=error_callback)

if __name__ == "__main__":
    # Path to the Ogg file
    file_path: str = './audio/audio.ogg'
    start_time: float = time.time()
    simulate_live_audio_stream(file_path)
    end_time: float = time.time()

    execution_time: float = end_time - start_time
    log.info(f"Execution time: {execution_time} seconds")