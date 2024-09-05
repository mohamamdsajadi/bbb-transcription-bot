# main.py
import os
import time
from typing import List, Optional

import ffmpeg # type: ignore

from prometheus_client import start_http_server

from stream_pipeline.grpc_server import GrpcServer

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ExecutionModule, ModuleOptions
from stream_pipeline.pipeline import Pipeline, ControllerMode, PipelinePhase, PipelineController

# from m_create_audio_package import CreateNsAudioPackage
import data
from extract_ogg import get_header_frames, split_ogg_data_into_frames, OggSFrame, calculate_frame_duration
import logger
# from m_stt_whisper import Whisper
# from asr_whisperx import Clean_Whisper_data, Local_Agreement, WhisperX_align, WhisperX_load_audio, WhisperX_transcribe
# from next_word import Next_Word_Prediction

from asr_faster_whisper import Create_Audio_Buffer, Load_audio, VAD, Faster_Whisper_transcribe, Local_Agreement

log = logger.setup_logging()

start_http_server(8000)

# CreateNsAudioPackage, Load_audio, VAD, Faster_Whisper_transcribe, Local_Agreement
controllers = [
    PipelineController(
        mode=ControllerMode.NOT_PARALLEL,
        max_workers=1,
        name="Create_Audio_Buffer",
        phases=[
            PipelinePhase(
                name="Create_Audio_Buffer",
                modules=[
                    Create_Audio_Buffer()
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
                    Load_audio(),
                    VAD(),
                    Faster_Whisper_transcribe(),
                    Local_Agreement(),
                ]
            )
        ]
    )
]

pipeline = Pipeline[data.AudioData](controllers, name="WhisperPipeline")

def callback(dp: DataPackage[data.AudioData]) -> None:
    if dp.data and dp.data.transcribed_segments:
        # log.info(f"Text: {dp.data.transcribed_text['words']}")
        processing_time = dp.total_time
        log.info(f"{processing_time:2f}:  {dp.data.confirmed_words} +++ {dp.data.unconfirmed_words}")
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

def simulate_live_audio_stream(file_path: str, sample_rate: int) -> None:
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
        print(frame_duration)

        audio_data: data.AudioData = data.AudioData(raw_audio_data=frame.raw_data, sample_rate=sample_rate)

        pipeline.execute(audio_data, instance, callback=callback, exit_callback=exit_callback, overflow_callback=overflow_callback, outdated_callback=outdated_callback, error_callback=error_callback)



def is_ogg_opus(file_path):
    """
    Check if a file is an Ogg Opus file using ffmpeg.
    """
    try:
        # Probe the file to get its metadata
        probe = ffmpeg.probe(file_path)
        # Find the audio stream
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        
        # If there are no audio streams, return False
        if not audio_streams:
            return False
        
        # Check if the codec_name is opus and the container format is ogg
        is_opus = any(stream['codec_name'] == 'opus' for stream in audio_streams)
        is_ogg = probe['format']['format_name'] == 'ogg'
        
        return is_opus and is_ogg
    except ffmpeg.Error as e:
        print(f"Error while probing file: {e}")
        return False

def get_sample_rate(file_path):
    """
    Get the sample rate of an Ogg Opus file using ffmpeg.
    """
    try:
        # Probe the file to get its metadata
        probe = ffmpeg.probe(file_path)
        # Find the audio stream
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        
        # If no audio stream found, return None
        if audio_stream is None:
            return None
        
        # Get the sample rate
        sample_rate = int(audio_stream['sample_rate'])
        return sample_rate
    except ffmpeg.Error as e:
        print(f"Error while probing file: {e}")
        return None

def convert_to_ogg_opus(input_file, output_file):
    """
    Convert any audio file to Ogg Opus format using ffmpeg.

    Args:
        input_file (str): The path to the input audio file.
        output_file (str): The path to the output Ogg Opus file.
    """
    try:
        # Use ffmpeg to convert the input file to Ogg Opus format
        ffmpeg.input(input_file).output(output_file, format='opus', acodec='libopus', audio_bitrate='64k').run()
        print(f"Conversion successful! Output file: {output_file}")
    except ffmpeg.Error as e:
        print(f"Error during conversion: {e.stderr.decode()}")

if __name__ == "__main__":
    # Path to the input audio file
    file_path = 'audio/audio-slow.ogg'  # Replace with your file path
    
    # Generate output file name with .ogg extension
    output_file = os.path.splitext(file_path)[0] + '.ogg'
    
    # Check if the file is already an Ogg Opus file
    if not is_ogg_opus(file_path):
        log.info(f"{file_path} is not an Ogg Opus file. Converting to Ogg Opus format...")
        
        # Check if the output file already exists
        if os.path.exists(output_file):
            raise FileExistsError(f"The output file '{output_file}' already exists. Aborting conversion.")
        
        # Convert the file to Ogg Opus format
        convert_to_ogg_opus(file_path, output_file)
    else:
        log.info(f"{file_path} is already an Ogg Opus file.")
    
    # Get sample rate of the resulting file (Ogg Opus)
    sample_rate = get_sample_rate(output_file)
    log.info(f"Sample rate: {sample_rate} Hz")
    
    # Simulate live audio stream (example usage)
    start_time = time.time()
    simulate_live_audio_stream(output_file, sample_rate)
    end_time = time.time()

    execution_time = end_time - start_time
    log.info(f"Execution time: {execution_time} seconds")