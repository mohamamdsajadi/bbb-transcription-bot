# main.py
import time
from typing import List, Optional


from stream_pipeline.grpc_server import GrpcServer

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ExecutionModule, ModuleOptions
from stream_pipeline.pipeline import Pipeline, ControllerMode, PipelinePhase, PipelineController

from m_create_audio_package import CreateNsAudioPackage
import data
from extract_ogg import get_header_frames, split_ogg_data_into_frames, OggSFrame, calculate_frame_duration
import logger
from m_stt_whisper import Whisper
from asr_whisperx import Clean_Whisper_data, Local_Agreement, WhisperX_align, WhisperX_load_audio, WhisperX_transcribe
from next_word import Next_Word_Prediction

log = logger.setup_logging()

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
        max_workers=1,
        name="MainProcessingController",
        phases=[
            PipelinePhase(
                name="WhisperPhase",
                modules=[
                    # Whisper()
                    WhisperX_load_audio(),
                    WhisperX_transcribe(),
                    WhisperX_align(),
                    Clean_Whisper_data(),
                    Local_Agreement(),
                    Next_Word_Prediction()
                ]
            )
        ]
    )
]

pipeline = Pipeline[data.AudioData](controllers, name="WhisperPipeline")

def callback(dp: DataPackage[data.AudioData]) -> None:
    if dp.data and dp.data.transcribed_text:
        # log.info(f"Text: {dp.data.transcribed_text['words']}")
        log.info(f"{dp.data.transcribed_text['confirmed_words']} +++ {dp.data.transcribed_text['unconfirmed_words']} +++ {dp.data.transcribed_text['next_word']}")
    pass
    
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