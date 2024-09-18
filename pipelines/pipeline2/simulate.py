# main.py
from dataclasses import dataclass
import json
import os
import pickle
import shutil
import subprocess
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

@dataclass
class Simulation_Pipeline:
    name: str
    prometheus_url: List[str]
    pipeline: List[PipelineController]

# CreateNsAudioPackage, Load_audio, VAD, Faster_Whisper_transcribe, Local_Agreement

simulation_pipeline_list = [
    Simulation_Pipeline(
        name = "p1",
        prometheus_url = [],
        pipeline = [
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
    ),
]




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

audio_extensions = [
    ".aac",    # Advanced Audio Codec
    ".ac3",    # Audio Codec 3
    ".aiff",   # Audio Interchange File Format
    ".aif",    # Audio Interchange File Format
    ".alac",   # Apple Lossless Audio Codec
    ".amr",    # Adaptive Multi-Rate audio codec
    ".ape",    # Monkey's Audio
    ".au",     # Sun Microsystems Audio
    ".dts",    # Digital Theater Systems audio
    ".eac3",   # Enhanced AC-3
    ".flac",   # Free Lossless Audio Codec
    ".m4a",    # MPEG-4 Audio (usually AAC)
    ".mka",    # Matroska Audio
    ".mp3",    # MPEG Layer 3
    ".ogg",    # Ogg Vorbis or Ogg Opus
    ".opus",   # Opus audio codec
    ".ra",     # RealAudio
    ".rm",     # RealMedia
    ".tta",    # True Audio codec
    ".voc",    # Creative Voice File
    ".wav",    # Waveform Audio File Format
    ".wma",    # Windows Media Audio
    ".wv",     # WavPack
    ".caf",    # Core Audio Format
    ".gsm",    # GSM 6.10 audio codec
    ".mp2",    # MPEG Layer 2 audio
    ".spx",    # Speex audio
    ".aob"     # Audio Object (used in DVD-Audio)
]

def main() -> None:
    input_folder  = './audio'
    output_folder = './simulate_results'
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for folder_name in os.listdir(input_folder):
        # Check if the file has a valid audio extension and skip non-audio files
        if not any(folder_name.endswith(ext) for ext in audio_extensions):
            print(f"Skipping non-audio file: {folder_name}")
            continue
        
        new_output_folder = os.path.join(output_folder, os.path.splitext(folder_name)[0])  # Create folder for each file
        if not os.path.exists(new_output_folder):
            os.makedirs(new_output_folder)

        input_file = os.path.join(input_folder, folder_name)
        output_file = os.path.join(new_output_folder, os.path.splitext(folder_name)[0] + '.ogg')
        
        # Skip if the output file already exists
        if os.path.exists(output_file):
            continue

        # Construct and run the ffmpeg command as before
        command = [
            'ffmpeg', '-i', input_file, '-c:a', 'libopus',
            '-frame_duration', '20', '-page_duration', '20000',
            '-vn', output_file
        ]

        try:
            subprocess.run(command, check=True)
            print(f"Converted: {folder_name} -> {output_file}")
        except Exception as e:
            print(f"Error processing file {folder_name}: {e}")
    
    for folder_name in os.listdir(output_folder):
        new_output_folder = os.path.join(output_folder, os.path.splitext(folder_name)[0])
        file_path = os.path.join(new_output_folder, folder_name + ".ogg")
        new_file_beginning = os.path.join(new_output_folder, folder_name)
        
        if not file_path.endswith(".ogg"):
            print(f"Skipping non-audio file: {folder_name}")
            continue
        

        for simulation_pipeline in simulation_pipeline_list:
            new_file_beginning_sumulation = new_file_beginning + f"_{simulation_pipeline.name}"
            
            if not os.path.exists(f"{new_file_beginning_sumulation}_simulation.pkl"):
                # create pipeline
                controllers = simulation_pipeline.pipeline
                pipeline = Pipeline[data.AudioData](controllers, name="WhisperPipeline")
                pipeline_id = pipeline.get_id()
                instance = pipeline.register_instance()

                def simulated_callback(raw_audio_data: bytes) -> None:
                    audio_data = data.AudioData(raw_audio_data=raw_audio_data)
                    pipeline.execute(
                                    audio_data, instance, 
                                    callback=callback,
                                    error_callback=error_callback
                                    )
                
                simulate_live_audio_stream(file_path, simulated_callback)
                time.sleep(5)

                with result_mutex:
                    data_list = [dat.data for dat in result if dat.data is not None]
                    with open(f"{new_file_beginning_sumulation}_simulation.pkl", 'wb') as file:
                        pickle.dump(data_list, file)

            if not os.path.exists(f"{new_file_beginning}_transcript.pkl"):
                transcript = transcribe_audio(file_path)
                with open(f"{new_file_beginning}_transcript.pkl", 'wb') as file:
                    pickle.dump(transcript, file)



            # Load the pkl file
            with open(f"{new_file_beginning_sumulation}_simulation.pkl", 'rb') as read_file:
                live_data: List[data.AudioData] = pickle.load(read_file) # type: ignore
                
            with open(f"{new_file_beginning}_transcript.pkl", 'rb') as read_file:
                transcript_words: List[data.Word] = pickle.load(read_file) # type: ignore

            # cw = Confirm_Words()
            # for live_dp in live_dps:
            #     if live_dp.data is not None:
            #         live_dp.data.confirmed_words = None
            #         live_dp.data.unconfirmed_words = None
            #     cw.execute(live_dp, DataPackageController(), DataPackagePhase(), DataPackageModule())

            live_dps: List[DataPackage[data.AudioData]] = [] # type: ignore
            for da in live_data:
                new_dp = DataPackage[data.AudioData]()
                new_dp.data=da
                live_dps.append(new_dp)

            if live_dps[-1].data is None:
                raise ValueError("No data found")
            live_words = live_dps[-1].data.confirmed_words
            
            if live_words is None:
                raise ValueError("No data found")
            stat_sensetive, stat_insensetive = stats(live_words, transcript_words)
            
            def save_stats(stats_sensetive, stats_insensetive) -> None:
                # Function to format statistics as JSON
                def stats_to_json(stat: Statistics) -> str:
                    return json.dumps({
                        "deletions": [{"word": word.word, "start": word.start, "end": word.end, "probability": word.probability} for word in stat.deletions],
                        "substitutions": [{"from": sub[0].word, "to": sub[1].word} for sub in stat.substitutions],
                        "insertions": [{"word": word.word, "start": word.start, "end": word.end, "probability": word.probability} for word in stat.insertions],
                        "wer": stat.wer,
                        "avg_delta_start": stat.avg_delta_start,
                        "avg_delta_end": stat.avg_delta_end
                    }, indent=4)

                # if file f"{new_file_beginning}_stats.txt" exists, delete it
                if os.path.exists(f"{new_file_beginning_sumulation}_stats.txt"):
                    os.remove(f"{new_file_beginning_sumulation}_stats.txt")

                # Writing the output to a file
                with open(f"{new_file_beginning_sumulation}_stats.txt", "w") as file:
                    file.write(f"-------------------------------------------------------------------\n")
                    file.write(f"File: {file_path}\n")
                    file.write(f"-------------------------------------------------------------------\n")
                    file.write(f"Statistics for case sensitive:\n")
                    file.write(f"Number of words missing in live (Deletions): {len(stats_sensetive.deletions)}\n")
                    file.write(f"Number of wrong words in live (Substitutions): {len(stats_sensetive.substitutions)}\n")
                    file.write(f"Number of extra words in live (Insertions): {len(stats_sensetive.insertions)}\n")
                    file.write(f"Average difference in start times: {stats_sensetive.avg_delta_start * 1000:.1f} milliseconds\n")
                    file.write(f"Average difference in end times: {stats_sensetive.avg_delta_end * 1000:.1f} milliseconds\n")
                    file.write(f"Word Error Rate (WER): {stats_sensetive.wer * 100:.1f}%\n")
                    file.write(f"-------------------------------------------------------------------\n")
                    file.write(f"Statistics without case sensitivity and symbols:\n")
                    file.write(f"Number of words missing in live (Deletions): {len(stats_insensetive.deletions)}\n")
                    file.write(f"Number of wrong words in live (Substitutions): {len(stats_insensetive.substitutions)}\n")
                    file.write(f"Number of extra words in live (Insertions): {len(stats_insensetive.insertions)}\n")
                    file.write(f"Average difference in start times: {stats_insensetive.avg_delta_start * 1000:.1f} milliseconds\n")
                    file.write(f"Average difference in end times: {stats_insensetive.avg_delta_end * 1000:.1f} milliseconds\n")
                    file.write(f"Word Error Rate (WER): {stats_insensetive.wer * 100:.1f}%\n")
                    file.write(f"-------------------------------------------------------------------\n")
                    file.write(f"-------------------------------------------------------------------\n")
                    file.write(f"Statistics as formatted JSON for sensitive case:\n")
                    file.write(stats_to_json(stats_sensetive) + "\n")
                    file.write(f"Statistics as formatted JSON for insensitive case:\n")
                    file.write(stats_to_json(stats_insensetive) + "\n")
                
            print(f"File: {folder_name}")
            save_stats(stat_sensetive, stat_insensetive)
    
if __name__ == "__main__":
    main()