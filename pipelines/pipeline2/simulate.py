# main.py
import pickle
import threading
import time
from typing import List
from prometheus_client import start_http_server

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.pipeline import Pipeline, ControllerMode, PipelinePhase, PipelineController

from m_convert_audio import Convert_Audio
from m_create_audio_buffer import Create_Audio_Buffer
from m_faster_whisper import Faster_Whisper_transcribe
from m_confirm_words import Confirm_Words
from m_rate_limiter import Rate_Limiter
from m_vad import VAD
import data
import logger
from simulate_live_audio_stream import Statistics, simulate_live_audio_stream, stats, transcribe_audio

log = logger.setup_logging()

start_http_server(8042)

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
        max_workers=3,
        queue_size=2,
        name="AudioPreprocessingController",
        phases=[
            PipelinePhase(
                name="VADPhase",
                modules=[
                    Convert_Audio(),
                    VAD(),
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
                    Faster_Whisper_transcribe(),
                ]
            )
        ]
    ),
    PipelineController(
        mode=ControllerMode.NOT_PARALLEL,
        max_workers=1,
        name="OutputController",
        phases=[
            PipelinePhase(
                name="OutputPhase",
                modules=[
                    Confirm_Words(),
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
        with result_mutex:
            result.append(dp)
            
        processing_time = dp.total_time

        if dp.data.confirmed_words is not None:
            only_words_c = [w.word for w in dp.data.confirmed_words]

        if dp.data.unconfirmed_words is not None:
            only_words_u = [w.word for w in dp.data.unconfirmed_words]
            
        if len(only_words_c) > 50:
            only_words_c = only_words_c[-50:]
            
        # print(f"({new_words}){only_words_c} ++ {only_words_u}")
        print(f"{processing_time:2f}:  {only_words_c} ")
    pass

def error_callback(dp: DataPackage[data.AudioData]) -> None:
    log.error("Pipeline error", extra={"data_package": dp})

instance = pipeline.register_instance()

def simulated_callback(raw_audio_data: bytes) -> None:
    audio_data = data.AudioData(raw_audio_data=raw_audio_data)
    pipeline.execute(
                    audio_data, instance, 
                    callback=callback,
                    error_callback=error_callback
                    )

def main() -> None:
    # Path to the input audio file
    file_path = 'audio/audio.ogg'  # Replace with your file path
    
    # Simulate live audio stream (example usage)
    simulate_live_audio_stream(file_path, simulated_callback)
    
    time.sleep(5)

    # Save the live transcription as a JSON
    with result_mutex:
        data_list: List[data.AudioData] = [dat.data for dat in result if dat.data is not None]
        with open('text.pkl', 'wb') as file:
            pickle.dump(data_list, file)

    # Load the JSON file
    with open('text.pkl', 'rb') as read_file:
        live_data: List[data.AudioData] = pickle.load(read_file) # type: ignore

    live_dps: List[DataPackage[data.AudioData]] = [] # type: ignore
    for da in live_data:
        new_dp = DataPackage[data.AudioData]()
        new_dp.data=da
        live_dps.append(new_dp)

    cw = Confirm_Words()
    for live_dp in live_dps:
        if live_dp.data is not None:
            live_dp.data.confirmed_words = None
            live_dp.data.unconfirmed_words = None
        cw.execute(live_dp, DataPackageController(), DataPackagePhase(), DataPackageModule())

    if live_dps[-1].data is None:
        raise ValueError("No data found")
    live_words = live_dps[-1].data.confirmed_words

    # safe transcript and result_tuple as json in a file
    transcript_words = transcribe_audio(file_path)
    if live_words is None:
        raise ValueError("No data found")
    stats_sensetive, stats_insensetive = stats(live_words, transcript_words)
    
    def print_stats(stat: Statistics) -> None:
        print(f"-------------------------------------------------------------------")
        print(f"Number of words missing in live (Deletions): {len(stat.deletions)}")
        print(f"Number of wrong words in live (Substitutions): {len(stat.substitutions)}")
        print(f"Number of extra words in live (Insertions): {len(stat.insertions)}")
        print(f"Average difference in start times: {stat.avg_delta_start * 1000:.1f} milliseconds")
        print(f"Average difference in end times: {stat.avg_delta_start * 1000:.1f} milliseconds")
        print(f"Word Error Rate (WER): {stat.wer * 100:.1f}%")
        print(f"-------------------------------------------------------------------")
        
    print_stats(stats_sensetive)
    print_stats(stats_insensetive)
    
if __name__ == "__main__":
    main()