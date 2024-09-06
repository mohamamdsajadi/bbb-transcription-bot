import time
import torch
import whisperx # type: ignore
from difflib import SequenceMatcher
from typing import Callable, List, Optional

from extract_ogg import OggSFrame, calculate_frame_duration, extract_id_header_frame, get_sample_rate, split_ogg_data_into_frames
import data
from stream_pipeline.data_package import DataPackage

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
        
        

def transcribe_audio(audio_file):
    device = "cuda" 
    batch_size = 32 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v3", device, compute_type=compute_type, asr_options={"word_timestamps": True})

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

    start = time.time()

    audio = whisperx.load_audio(audio_file)

    result = model.transcribe(audio, batch_size=batch_size)
    # print(result) # before alignment

    # Trim result["segments"] to 10 items
    result["segments"] = result["segments"][:10]

    # delete model if low on GPU resources
    import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # delete model if low on GPU resources
    import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    print(result["segments"])
    return result["segments"]


def create_live_transcription_tuple(live_transcription: List[DataPackage[data.AudioData]]):
    """Create a tuple of live transcription"""
    """(0, ['confirmed', 'words'], ['unconfirmed', 'words'])"""

    first_start_time = live_transcription[0].start_time

    return_tuple = []
    for dp in live_transcription:
        if dp.data and dp.data.confirmed_words and dp.data.unconfirmed_words:
            relative_time = dp.end_time - first_start_time
            return_tuple.append((relative_time, dp.data.confirmed_words, dp.data.unconfirmed_words))

    return return_tuple


def calculate_avg_time_difference(live_transcription, transcript, offset=5):

    def _find_segments(relative_time, transcript_segments, offset=10):
        """Finds segments within a given time range."""
        segments = []
        for segment in transcript_segments:
            start = segment["start"]
            end = segment["end"]
            
            # Check if relative_time is within the range or within the offset
            if start - offset <= relative_time <= end + offset:
                segments.append(segment)
                
        return segments

    def _similarity_ratio(word1, word2):
        """Calculate the similarity ratio between two words using SequenceMatcher."""
        return SequenceMatcher(None, word1, word2).ratio()

    # Initialize variables
    total_time_difference_confirmed = 0.0
    confirmed_word_count = 0

    # Loop through the live transcription data
    for index, transcription in enumerate(live_transcription):
        # Simulate real-time processing
        # time.sleep(1)
        
        time_finished, confirmed, unconfirmed = transcription
        current_time = time_finished
        
        # Find relevant segments around the current time
        relevant_segments = _find_segments(current_time, transcript, offset=offset)
        
        # Find spoken words in the segments that match the confirmed words
        for segment in relevant_segments:
            for word_info in segment["words"]:
                word = word_info["word"]
                
                # Check if word is already used
                if word_info.get("used"):
                    continue

                if word_info.get("start") is None or word_info.get("end") is None:
                    continue
                
                # Find the confirmed word with the highest similarity to the spoken word
                max_similarity = 0
                most_similar_word = None
                
                for confirmed_word in confirmed:
                    similarity = _similarity_ratio(word, confirmed_word)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_word = confirmed_word
                
                # If a similar word is found and not yet used, calculate the time difference
                if max_similarity >= 0.9:
                    time_difference = current_time - word_info["end"]
                    
                    # Check if the time difference is negative or larger than 5 seconds
                    if time_difference < 0 or time_difference > offset:
                        continue  # Skip this word since it's not a match
                    
                    # Valid confirmed word processing
                    confirmed_word_count += 1
                    total_time_difference_confirmed += time_difference
                    # if max_similarity < 1:
                    #     print(f"Found similar confirmed word '{word}'=='{most_similar_word}' at time {current_time} with a time difference of {time_difference:.2f} seconds")
                    
                    # Mark word as used
                    word_info["used"] = True
                    
    avg = -1.0
    if confirmed_word_count > 0:
        avg = total_time_difference_confirmed / confirmed_word_count
    
    
    # # List of all words which are still unused
    # unused_words = []
    # for segment in transcript:
    #     for word_info in segment["words"]:
    #         if not word_info.get("used"):
    #             unused_words.append((word_info["word"], word_info["start"]))
                
    # print("Unused words in the transcript:", len(unused_words), unused_words)
    
    return avg

# # Calculate the average time difference for confirmed words
# avg = calculate_avg_time_difference(live_transcription, transcript)
# print(f"Average time difference for confirmed words: {avg:.2f} seconds")
