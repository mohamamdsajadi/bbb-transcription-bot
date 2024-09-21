from dataclasses import dataclass
import difflib
import statistics
import time
import unicodedata
import torch
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional, Tuple, Union

from faster_whisper import WhisperModel, BatchedInferencePipeline # type: ignore
import torch

from ogg import Ogg_OPUS_Audio, OggS_Page, calculate_page_duration
import data
from stream_pipeline.data_package import DataPackage

def simulate_live_audio_stream(file_path: str, callback: Callable[[bytes], None]) -> Tuple[float, float]:
    with open(file_path, 'rb') as file:
        ogg_bytes: bytes = file.read()

    audio = Ogg_OPUS_Audio(ogg_bytes)
    id_header_page = audio.id_header
    if id_header_page is None:
        raise ValueError("No ID header page found")
    sample_rate = id_header_page.input_sample_rate

    start = time.time()

    previous_granule_position: Optional[int] = None
    for page_index, page in enumerate(audio.pages):
        current_granule_position: int = page.granule_position
        page_duration: float = calculate_page_duration(current_granule_position, previous_granule_position, sample_rate)
        previous_granule_position = current_granule_position

        callback(page.raw_data)

        # Sleep to simulate real-time audio playback
        time.sleep(page_duration)
    
    end = time.time()
    
    return (start, end)
        
        

def transcribe_audio(audio_path: str, model_path: Optional[str]) -> List[data.Word]:
    # Configuration for the Whisper model
    model_size = "large-v3"
    compute_type = "float16"  # Options: "float16" or "int8"
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the Whisper model
    print(f"Loading Whisper model: '{model_size}' on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root=model_path)
    batched_model = BatchedInferencePipeline(model=model)
    print("Whisper model loaded successfully!")

    # Transcribe the audio using the model
    segments, info = batched_model.transcribe(audio_path, batch_size=batch_size, word_timestamps=True)

    # Convert segments to TextSegment objects
    result = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                w = data.Word(
                    word=word.word,
                    start=word.start,
                    end=word.end,
                    probability=word.probability
                )
                result.append(w)

    return result


@dataclass
class Statistics:
    deletions: List[data.Word]
    substitutions: List[Tuple[data.Word, data.Word]]
    insertions: List[data.Word]
    wer: float
    avg_delta_start: float
    avg_delta_end: float

def compute_statistics(
    live: List[data.Word], 
    transcript: List[data.Word]
) -> Statistics:
    
    if len(live) == 0:
        raise ValueError("The 'live' list is empty")
    if len(transcript) == 0:
        raise ValueError("The 'transcript' list is empty")

    # Variables with types
    last_live_word: data.Word = live[-1]

    # Only use transcript until the last live word
    new_transcript: List[data.Word] = [word for word in transcript if word.end <= last_live_word.end]

    # Extract word strings from the Word objects, stripping leading/trailing spaces
    live_words: List[str] = [w.word.strip() for w in live]
    transcript_words: List[str] = [w.word.strip() for w in new_transcript]

    # Create a SequenceMatcher object to compare the two sequences
    sm: difflib.SequenceMatcher = difflib.SequenceMatcher(None, transcript_words, live_words)

    # Lists to store deletions, substitutions, and insertions
    deletion_list: List[data.Word] = []
    substitution_list: List[Tuple[data.Word, data.Word]] = []
    insertion_list: List[data.Word] = []

    # Lists to store time differences for matching words
    delta_starts: List[float] = []
    delta_ends: List[float] = []

    # Process the opcodes to align the sequences and identify operations
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            # Words match; calculate time differences
            for idx_transcript, idx_live in zip(range(i1, i2), range(j1, j2)):
                word_transcript = new_transcript[idx_transcript]
                word_live = live[idx_live]
                delta_start = word_live.start - word_transcript.start
                delta_end = word_live.end - word_transcript.end
                delta_starts.append(delta_start)
                delta_ends.append(delta_end)
        elif tag == 'replace':
            # Substitution
            substitutions: List[Tuple[data.Word, data.Word]] = [
                (new_transcript[idx], live[idx2]) for idx, idx2 in zip(range(i1, i2), range(j1, j2))
            ]
            substitution_list.extend(substitutions)
        elif tag == 'delete':
            # Deletion
            deletions: List[data.Word] = new_transcript[i1:i2]
            deletion_list.extend(deletions)
        elif tag == 'insert':
            # Insertion
            insertions: List[data.Word] = live[j1:j2]
            insertion_list.extend(insertions)

    # Compute Word Error Rate (WER)
    S: int = len(substitution_list)  # Number of substitutions
    D: int = len(deletion_list)  # Number of deletions
    I: int = len(insertion_list)  # Number of insertions
    N: int = len(transcript_words) + S + D # Correct words + substitutions + deletions
    WER: float = (S + D + I) / N if N > 0 else 0

    # Compute average differences in start and end times (in seconds)
    avg_delta_start: float = sum(abs(ds) for ds in delta_starts) / len(delta_starts) if delta_starts else 0
    avg_delta_end: float = sum(abs(de) for de in delta_ends) / len(delta_ends) if delta_ends else 0

    # Return the statistics as a dataclass instance
    return Statistics(
        deletions=deletion_list,
        substitutions=substitution_list,
        insertions=insertion_list,
        wer=WER,
        avg_delta_start=avg_delta_start,
        avg_delta_end=avg_delta_end,
    )


def _is_similar(word1: str, word2: str, max_diff_percantage: float = -1.0) -> bool:
    def similarity_difflib(wort1: str, wort2: str) -> float:
        matcher = difflib.SequenceMatcher(None, wort1, wort2)
        return matcher.ratio()
    
    # Lowercase the words
    word1_l = word1.lower()
    word2_l = word2.lower()
    
    # Remove symbols and punctuation characters
    def remove_symbols(word: str) -> str:
        # Filter out characters classified as punctuation or symbols
        return ''.join(
            char for char in word 
            if not unicodedata.category(char).startswith(('P', 'S'))
        )
    
    word1_clean = remove_symbols(word1_l)
    word2_clean = remove_symbols(word2_l)

    if max_diff_percantage == -1.0:
        return word1_clean == word2_clean
    
    diff = similarity_difflib(word1_clean, word2_clean)

    return diff >= max_diff_percantage

def time_difference(live_dps: List[DataPackage[data.AudioData]], transcript: List[data.Word], offset: float = 0.4) -> float:
    # get the difference between live_dps[n].data.confirmed_words and live_dps[n+1].data.confirmed_words
    # def get_diff(cw1: Optional[List[data.Word]], cw2: Optional[List[data.Word]]) -> List[data.Word]:
    #     if cw1 is None or cw2 is None or len(cw1) == 0 or len(cw2) == 0:
    #         return []
    #     last_word_time = cw1[-1].end
    #     diff_words = [word for word in cw2 if word.start >= last_word_time]
    #     return diff_words
    
    # List of transcript words to check if the word is found in the transcript
    transcript_list: List[data.Word] = transcript.copy()
    
    diff: List[float] = []
    # last_live_dp: Optional[DataPackage[data.AudioData]] = None
    for i, live_dp in enumerate(live_dps):
        if live_dp.data is None:
            continue
        
        if live_dp.data is not None:
            # diff_words = get_diff(last_live_dp.data.confirmed_words, live_dp.data.confirmed_words)
            
            live_words_confimed = live_dp.data.confirmed_words
            
            if live_words_confimed is not None:
                
                # only take the word if they are at the same time point +- offset in the transcript
                correct_diff_words = []
                to_remove = []
                for tword in transcript_list:
                    word_in_transcript = next((lw for lw in live_words_confimed if lw.start - offset <= tword.start <= lw.end + offset and _is_similar(lw.word, tword.word, 0.7)), None)
                    if word_in_transcript is not None:
                        correct_diff_words.append(tword)
                        to_remove.append(tword)
                        
                # remove the words from the transcript list
                for tword in to_remove:
                    transcript_list.remove(tword)
                
                
                
                diff_words_end = [word.end for word in correct_diff_words]
                
                if live_dp.data.audio_buffer_start_after is None or live_dp.data.audio_buffer_time is None:
                    continue
                
                processing_time = live_dp.end_time - live_dp.start_time
                current_audio_buffer_time = live_dp.data.audio_buffer_start_after + live_dp.data.audio_buffer_time
                output_time = current_audio_buffer_time + processing_time
                
                # current_audio_buffer_time - diff_words_end
                diff_words_time = [output_time - end for end in diff_words_end]
                diff.extend(diff_words_time)
                print(f"{output_time}")
                
            
        last_live_dp = live_dp
        print(f"Timedifference: {len(diff)}/{len(transcript_list)} found")
        
    return statistics.mean(diff)


# list[DataPackage[AudioData]]
def stats(live_dps: List[DataPackage[data.AudioData]], transcript: List[data.Word]) -> Tuple[Statistics, Statistics, float]:

    avg_time_difference = time_difference(live_dps, transcript)

    if live_dps[-1].data is None:
        raise ValueError("No data found")
    live_words = live_dps[-1].data.confirmed_words

    if live_words is None:
        raise ValueError("No data found")
    diff = compute_statistics(live_words, transcript)

    def to_lower_no_symbols(word: str) -> str:
        word_l = word.lower()
        
        # Remove symbols and punctuation characters
        def remove_symbols(word: str) -> str:
            # Filter out characters classified as punctuation or symbols
            return ''.join(
                char for char in word 
                if not unicodedata.category(char).startswith(('P', 'S'))
            )
        
        word_clean = remove_symbols(word_l)
        return word_clean
    
    live_clean = [
        data.Word(
            to_lower_no_symbols(word.word),
            word.start,
            word.end,
            word.probability
        )
        for word in live_words
    ]
    transcript_clean = [
        data.Word(
            to_lower_no_symbols(word.word),
            word.start,
            word.end,
            word.probability
        )
        for word in transcript
    ]

    diff2 = compute_statistics(live_clean, transcript_clean)

    return diff, diff2, avg_time_difference
