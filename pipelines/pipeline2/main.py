# main.py
import os
import threading
import time
from typing import List, Optional
import json

import ffmpeg # type: ignore
from prometheus_client import start_http_server
from flask import Flask

from stream_pipeline.data_package import DataPackage
from stream_pipeline.pipeline import Pipeline, ControllerMode, PipelinePhase, PipelineController

from Config import load_settings
from StreamServer import Server
from Client import Client
from m_convert_audio import Convert_Audio
from m_create_audio_buffer import Create_Audio_Buffer
from m_faster_whisper import Faster_Whisper_transcribe
from m_confirm_words import Confirm_Words
from m_rate_limiter import Rate_Limiter
from m_vad import VAD
import data
import logger
# from simulate_live_audio_stream import calculate_statistics, create_live_transcription_tuple, simulate_live_audio_stream, transcribe_audio

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
        max_workers=5,
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
# def callback(dp: DataPackage[data.AudioData]) -> None:
#     if dp.data and dp.data.transcribed_segments:
#         # log.info(f"Text: {dp.data.transcribed_text['words']}")
#         processing_time = dp.total_time
#         log.info(f"{processing_time:2f}:  {dp.data.confirmed_words} +++ {dp.data.unconfirmed_words}")
#         with result_mutex:
#             result.append(dp)
#     pass
    
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

# def simulated_callback(raw_audio_data: bytes) -> None:
#     audio_data = data.AudioData(raw_audio_data=raw_audio_data)
#     pipeline.execute(
#                     audio_data, instance, 
#                     callback=callback, 
#                     exit_callback=exit_callback, 
#                     overflow_callback=overflow_callback, 
#                     outdated_callback=outdated_callback, 
#                     error_callback=error_callback
#                     )

# dev main():
#     # Path to the input audio file
#     file_path = 'audio/audio_output.ogg'  # Replace with your file path
    
#     # Simulate live audio stream (example usage)
#     start_simulation_time, end_simulation_time = simulate_live_audio_stream(file_path, simulated_callback)
    
#     time.sleep(5)

#     # Save the live transcription as a JSON
#     with result_mutex:
#         result_tuple = create_live_transcription_tuple(result, start_simulation_time)
#         with open('live_transcript.json', 'w') as f:
#             json.dump(result_tuple, f)

#     # Load the JSON file
#     with open('live_transcript.json', 'r') as f:
#         loaded_data = json.load(f)
#     result_tuple = tuple(loaded_data)

#     # safe transcript and result_tuple as json in a file
#     transcript = transcribe_audio(file_path)
#     with open('transcript.json', 'w') as f:
#         json.dump(transcript, f)
#     uncon, con = calculate_statistics(result_tuple, transcript, 5)

#     execution_time = end_simulation_time - start_simulation_time
#     print(f"Execution time: {execution_time} seconds")
#     print(f"Average transcription time: {uncon[0]} seconds")
#     print(f"Min time difference: {uncon[1]} seconds")
#     print(f"Max time difference: {uncon[2]} seconds")
#     print(f"Median: {uncon[3]} seconds")
#     print(f"Standard deviation: {uncon[4]} seconds")
#     print("----------------------------------------------------")
#     print(f"Average confirmation time: {con[0]} seconds")
#     print(f"Min time difference: {con[1]} seconds")
#     print(f"Max time difference: {con[2]} seconds")
#     print(f"Median: {con[3]} seconds")
#     print(f"Standard deviation: {con[4]} seconds")

# Health check http sever
app = Flask(__name__)
STATUS = "stopped" # starting, running, stopping, stopped
@app.route('/health', methods=['GET'])
def healthcheck():
    global STATUS
    print(STATUS)
    if STATUS == "running":
        return STATUS, 200
    else:
        return STATUS, 503

def main():
    global STATUS
    STATUS = "starting"
    
    settings = load_settings()

    # Start the health http-server (flask) in a new thread.
    webserverthread = threading.Thread(target=app.run, kwargs={'debug': False, 'host': settings["HOST"], 'port': settings["HEALTH_CHECK_PORT"]})
    webserverthread.daemon = True  # This will ensure the thread stops when the main thread exits
    webserverthread.start()

    client_dict = {}        # Dictionary with all connected clients
    client_dict_mutex = threading.Lock() # Mutex to lock the client_dict

    # Create server
    srv = Server(settings["HOST"], settings["TCPPORT"], settings["UDPPORT"], settings["SECRET_TOKEN"], 4096, 5, 10, 1024, settings["EXTERNALHOST"])

    # Handle new connections and disconnections, timeouts and messages
    def OnConnected(c):
        print(f"Connected by {c.tcp_address()}")

        # Create new client
        newclient = Client(c)
        newclient._instance = instance # TODO
        # newclient._instance = pipeline.register_instance()
        with client_dict_mutex:
            client_dict[c] = newclient

        # Handle disconnections
        def ondisconnedted(c):
            print(f"Disconnected by {c.tcp_address()}")
            # Remove client from client_dict
            with client_dict_mutex:
                if c in client_dict:
                    del client_dict[c]
        c.on_disconnected(ondisconnedted)

        # Handle timeouts
        def ontimeout(c):
            print(f"Timeout by {c.tcp_address()}")
            # Remove client from client_dict
            with client_dict_mutex:
                if c in client_dict:
                    del client_dict[c]
        c.on_timeout(ontimeout)

        # Handle messages
        def onmsg(c, recv_data):
            # print(f"UDP from: {c.tcp_address()}")
            with client_dict_mutex:
                if not c in client_dict:
                    print(f"Client {c.tcp_address()} not in list!")
                    return
                client = client_dict[c]
            
                audio_data = data.AudioData(raw_audio_data=recv_data)
                pipeline.execute(
                                audio_data, 
                                client._instance, 
                                callback=callback, 
                                exit_callback=exit_callback, 
                                overflow_callback=overflow_callback, 
                                outdated_callback=outdated_callback, 
                                error_callback=error_callback
                                )
                

        c.on_udp_message(onmsg)
    srv.on_connected(OnConnected)

    def callback(dp: DataPackage[data.AudioData]) -> None:
        if dp.data and dp.data.confirmed_words is not None and dp.data.unconfirmed_words is not None:
            # log.info(f"Text: {dp.data.transcribed_text['words']}")
            processing_time = dp.total_time
            log.info(f"{processing_time:2f}:  {dp.data.confirmed_words} +++ {dp.data.unconfirmed_words}")
            # log.info(f"{processing_time:2f}: cleaned_words:  {dp.data.transcribed_segments}")
            
            
            # put dp.data.confirmed_words together with space
            text = ""
            for word in dp.data.confirmed_words:
                # if there is a . in this word add \n behind it
                # if "." in word.word:
                #     text += word.word + "\n"
                # else:
                text += word.word + " "
            for word in dp.data.unconfirmed_words:
                text += word.word + " "
            
            # get client
            with client_dict_mutex:
                for c in client_dict:
                    client = client_dict[c]
                    if client._instance == dp.pipeline_instance_id:
                        client.send(str.encode(text))

    # Start server
    print(f"Starting server: {settings['HOST']}:{settings['TCPPORT']}...")
    srv.start()
    print("Ready to transcribe. Press Ctrl+C to stop.")

    STATUS = "running"

    # Wait until stopped by Strg + C
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass

    # Stop server
    STATUS = "stopping"
    print("Stopping server...")
    srv.stop()
    print("Server stopped")
    STATUS = "stopped"
    
if __name__ == "__main__":
    main()