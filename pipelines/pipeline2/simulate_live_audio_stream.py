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
        
        

def transcribe_audio(audio_path: str) -> List[data.Word]:
    # Configuration for the Whisper model
    model_size = "large-v3"
    compute_type = "float16"  # Options: "float16" or "int8"
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the Whisper model
    print(f"Loading Whisper model: '{model_size}' on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
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
    N: int = len(transcript_words)  # Total words in transcript (reference)
    S: int = len(substitution_list)  # Number of substitutions
    D: int = len(deletion_list)  # Number of deletions
    I: int = len(insertion_list)  # Number of insertions
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

def stats(live: List[data.Word], transcript: List[data.Word]) -> Tuple[Statistics, Statistics]:
    diff = compute_statistics(live, transcript)

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
        for word in live
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

    return diff, diff2
