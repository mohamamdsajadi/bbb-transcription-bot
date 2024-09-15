# m_local_agreement.py
from typing import List, Tuple
import unicodedata

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ModuleOptions

import logger
import data

log = logger.get_logger()

class Confirm_Words(Module):
    def __init__(self) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=False,
                timeout=5,
            ),
            name="Confirm_Words"
        )
        self.max_confirmed_words = 0
        self.confirmed: List[data.Word] = []  # Buffer to store committed words

        self.confirmed_end_time: float = 0.0
        
        self.confirm_if_older_then: float = 2.0 # Confirm words if they are older than this value in seconds

    def init_module(self) -> None:
        pass

    def is_similar(self, word1: str, word2: str, max_diff_chars: int = 1) -> bool:
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
        
        # Calculate the number of different characters between word1 and word2
        diff_chars = sum(1 for a, b in zip(word1_clean, word2_clean) if a != b) + abs(len(word1_clean) - len(word2_clean))
        
        # Return True if the number of different characters is within the allowed maximum
        return diff_chars <= max_diff_chars

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not dp.data or dp.data.transcribed_segments is None:
            raise Exception("No transcribed words found")
        
        # Collect new words from the transcribed segments
        new_words: List[data.Word] = []
        for segment in dp.data.transcribed_segments:
            if segment.words is None:
                continue
            new_words.extend(segment.words)
            
        if len(new_words) == 0:
            dp.data.confirmed_words = self.confirmed.copy()
            dp.data.unconfirmed_words = []
            return
        
        newest_word_end_time = new_words[-1].end

        # 1. Split in confirmed, unconfirmed and new words
        new_confirmed: List[data.Word] = []
        new_unconfirmed: List[data.Word] = []
        for new_word in new_words:
            if new_word.start < self.confirmed_end_time:
                new_confirmed.append(new_word)
            else:
                new_unconfirmed.append(new_word)

        # 2. Check each new_unconfirmed word if it's older than confirm_if_older_then seconds
        words_to_confirm = []
        for new_word in new_unconfirmed:
            if newest_word_end_time - new_word.end >= self.confirm_if_older_then:
                self.confirmed.append(new_word)
                words_to_confirm.append(new_word)

        # Remove confirmed words from unconfirmed list
        for word in words_to_confirm:
            new_unconfirmed.remove(word)

        # Find words which are in new_confirmed and not in confirmed. Use simular
        for new_word in list(reversed(new_confirmed)):
            found = False
            for confirmed_word in list(reversed(list(self.confirmed))):
                if self.is_similar(confirmed_word.word, new_word.word):
                    found = True
                    break
            if not found:
                self.confirmed.append(new_word)

        # Modify confirmed words if they are in the same time range -+offset and are similar
        # start from the end of the list
        # for a, new_word in enumerate(list(reversed(list(new_confirmed)))):
        #     for b, confirmed_word in enumerate(list(reversed(list(self.confirmed)))):
        #         if abs(confirmed_word.start - new_word.start) <= 0.1 and abs(confirmed_word.end - new_word.end) <= 0.1:
        #             if self.is_similar(confirmed_word.word, new_word.word):
        #                 self.confirmed[-b] = new_word

        # sort confirmed words by start time
        self.confirmed = sorted(self.confirmed, key=lambda x: x.start)

        # Ensure that the number of confirmed words does not exceed the max_confirmed_words limit
        if len(self.confirmed) > self.max_confirmed_words:
            self.confirmed = self.confirmed[-self.max_confirmed_words:]

        if len(self.confirmed) > 0:
            self.confirmed_end_time = self.confirmed[-1].end
            
        if len(self.confirmed) > self.max_confirmed_words:
            self.confirmed = self.confirmed[-self.max_confirmed_words:]
        
        # Update data package confirmed and unconfirmed words
        dp.data.confirmed_words = self.confirmed.copy()
        dp.data.unconfirmed_words = new_unconfirmed
