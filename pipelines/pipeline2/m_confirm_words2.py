# m_local_agreement.py
import difflib
import sys
from typing import List, Optional, TextIO
from dataclasses import dataclass
import sys
import unicodedata

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ModuleOptions

import logger
import data

log = logger.get_logger()









class HypothesisBuffer:
    def __init__(self) -> None:
        self.commited_in_buffer: List[data.Word] = []
        self.buffer: List[data.Word] = []
        self.new: List[data.Word] = []

        self.last_commited_time: float = 0
        self.last_commited_word: Optional[str] = None

    def insert(self, new: List[data.Word], offset: float) -> None:
        new = [data.Word(w.word, w.start + offset, w.end + offset, w.probability) for w in new]
        self.new = [w for w in new if w.start > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            first_word = self.new[0]
            # Adjusted threshold to handle larger buffer jumps
            if abs(first_word.start - self.last_commited_time) < 5:  # Increased from 1 to 5 seconds
                if self.commited_in_buffer:
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):
                        c = " ".join([self.commited_in_buffer[-j].word for j in range(1, i + 1)][::-1])
                        tail = " ".join(self.new[j - 1].word for j in range(1, i + 1))
                        if c == tail:
                            # print(f"Removing last {i} words due to overlap:")
                            for j in range(i):
                                removed_word = self.new.pop(0)
                                # print(f"\tRemoved word: {removed_word.word}")
                            break

    def flush(self) -> List[data.Word]:
        commit: List[data.Word] = []
        while self.new:
            new_word = self.new[0]

            if len(self.buffer) == 0:
                break

            if new_word.word == self.buffer[0].word:
                commit.append(new_word)
                self.last_commited_word = new_word.word
                self.last_commited_time = new_word.end
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)

        # Logging the committed words
        # if commit:
            # print("Committed words:", [w.word for w in commit])
        return commit

    def pop_commited(self, time: float) -> None:
        initial_len = len(self.commited_in_buffer)
        while self.commited_in_buffer and self.commited_in_buffer[0].end <= time:
            removed_word = self.commited_in_buffer.pop(0)
            # print(f"Discarded old committed word: {removed_word.word}")
        # if initial_len != len(self.commited_in_buffer):
            # print(f"Updated committed_in_buffer, new length: {len(self.commited_in_buffer)}")

    def incomplete(self) -> List[data.Word]:
        return self.buffer









class Confirm_Words(Module):
    def __init__(self) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=False,
                timeout=5,
            ),
            name="Confirm_Words"
        )
        self.buffer = HypothesisBuffer()
        self.last_audio_buffer_start_after = 0.0  # Initialize to zero

    def init_module(self) -> None:
        pass

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not dp.data or dp.data.transcribed_segments is None:
            raise Exception("No transcribed words found")
        if dp.data.audio_buffer_start_after is None:
            raise Exception("No audio buffer start time found")
        if dp.data.audio_buffer_time is None:
            raise Exception("No audio buffer time found")
        
        audio_buffer_start_after = dp.data.audio_buffer_start_after
        audio_buffer_time = dp.data.audio_buffer_time

        # If the audio buffer has moved forward, reset the HypothesisBuffer
        if audio_buffer_start_after > self.last_audio_buffer_start_after:
            # Update last_audio_buffer_start_after
            self.last_audio_buffer_start_after = audio_buffer_start_after

            # Remove committed words that are no longer relevant
            self.buffer.pop_commited(audio_buffer_start_after)

            # Reset last_commited_time and last_commited_word
            self.buffer.last_commited_time = audio_buffer_start_after
            self.buffer.last_commited_word = None

            # Clear buffer and new
            self.buffer.buffer = []
            self.buffer.new = []

        # Collect new words from the transcribed segments
        new_words: List[data.Word] = []
        for segment in dp.data.transcribed_segments:
            if segment.words is None:
                continue
            new_words.extend(segment.words)

        # Insert new words into the HypothesisBuffer with the appropriate offset
        self.buffer.insert(new_words, audio_buffer_start_after)

        # Flush the buffer to get confirmed words
        committed_words = self.buffer.flush()

        # Set confirmed and unconfirmed words in the data package
        dp.data.confirmed_words = committed_words
        dp.data.unconfirmed_words = self.buffer.incomplete()


