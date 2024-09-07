# main.py
import os
import threading
import time
from typing import List, Optional
import json

import ffmpeg # type: ignore
from prometheus_client import start_http_server

from stream_pipeline.data_package import DataPackage
from stream_pipeline.pipeline import Pipeline, ControllerMode, PipelinePhase, PipelineController

from m_convert_audio import Convert_Audio
from m_create_audio_buffer import Create_Audio_Buffer
from m_faster_whisper import Faster_Whisper_transcribe
from m_local_agreement import Local_Agreement
from m_rate_limiter import Rate_Limiter
from m_remove_halicunation import Remove_Hallucination
from m_vad import VAD
import data
import logger
from simulate_live_audio_stream import calculate_statistics, create_live_transcription_tuple, simulate_live_audio_stream, transcribe_audio

log = logger.setup_logging()

start_http_server(8000)

# CreateNsAudioPackage, Load_audio, VAD, Faster_Whisper_transcribe, Local_Agreement
controllers = [
    PipelineController(
        mode=ControllerMode.NOT_PARALLEL,
        max_workers=1,
        queue_size=10,
        name="Create_Audio_Buffer",
        phases=[
            PipelinePhase(
                name="Create_Audio_Buffer",
                modules=[
                    Create_Audio_Buffer(),
                    Rate_Limiter(),
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
                    Convert_Audio(),
                    VAD(),
                    Faster_Whisper_transcribe(),
                    Remove_Hallucination(),
                    Local_Agreement(),
                ]
            )
        ]
    )
]

pipeline = Pipeline[data.AudioData](controllers, name="WhisperPipeline")

result: List[DataPackage[data.AudioData]] = []
result_mutex = threading.Lock()
def callback(dp: DataPackage[data.AudioData]) -> None:
    if dp.data and dp.data.transcribed_segments:
        # log.info(f"Text: {dp.data.transcribed_text['words']}")
        processing_time = dp.total_time
        log.info(f"{processing_time:2f}:  {dp.data.confirmed_words} +++ {dp.data.unconfirmed_words}")
        with result_mutex:
            result.append(dp)
    pass
    
def exit_callback(dp: DataPackage[data.AudioData]) -> None:
    # log.info(f"Exit: {dp.controllers[-1].phases[-1].modules[-1].message}")
    pass

def overflow_callback(dp: DataPackage[data.AudioData]) -> None:
    log.info("Overflow")
    pass

def outdated_callback(dp: DataPackage[data.AudioData]) -> None:
    log.info("Outdated", extra={"data_package": dp})

def error_callback(dp: DataPackage[data.AudioData]) -> None:
    log.error("Pipeline error", extra={"data_package": dp})

instance = pipeline.register_instance()

def simulated_callback(raw_audio_data: bytes) -> None:
    audio_data = data.AudioData(raw_audio_data=raw_audio_data)
    pipeline.execute(
                    audio_data, instance, 
                    callback=callback, 
                    exit_callback=exit_callback, 
                    overflow_callback=overflow_callback, 
                    outdated_callback=outdated_callback, 
                    error_callback=error_callback
                    )

if __name__ == "__main__":
    # Path to the input audio file
    file_path = 'audio/audio.ogg'  # Replace with your file path
    
    # Simulate live audio stream (example usage)
    start_simulation_time, end_simulation_time = simulate_live_audio_stream(file_path, simulated_callback)
    
    time.sleep(5)

    # Save the live transcription as a JSON
    with result_mutex:
        result_tuple = create_live_transcription_tuple(result, start_simulation_time)
        with open('live_transcript.json', 'w') as f:
            json.dump(result_tuple, f)

    # Load the JSON file
    with open('live_transcript.json', 'r') as f:
        loaded_data = json.load(f)
    result_tuple = tuple(loaded_data)

    # safe transcript and result_tuple as json in a file
    transcript = transcribe_audio(file_path)
    uncon, con = calculate_statistics(result_tuple, transcript, 5)

    execution_time = end_simulation_time - start_simulation_time
    print(f"Execution time: {execution_time} seconds")
    print(f"Average transcription time: {uncon[0]} seconds")
    print(f"Min time difference: {uncon[1]} seconds")
    print(f"Max time difference: {uncon[2]} seconds")
    print(f"Median: {uncon[3]} seconds")
    print(f"Standard deviation: {uncon[4]} seconds")
    print("----------------------------------------------------")
    print(f"Average confirmation time: {con[0]} seconds")
    print(f"Min time difference: {con[1]} seconds")
    print(f"Max time difference: {con[2]} seconds")
    print(f"Median: {con[3]} seconds")
    print(f"Standard deviation: {con[4]} seconds")
