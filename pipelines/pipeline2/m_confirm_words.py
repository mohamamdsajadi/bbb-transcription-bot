# m_local_agreement.py
from typing import List, Optional, Tuple
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

    def is_similar(self, word1: str, word2: str, max_diff_chars: int = -1) -> bool:
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

        if max_diff_chars == -1:
            return word1_clean == word2_clean
        
        # Calculate the number of different characters between word1 and word2
        diff_chars = sum(1 for a, b in zip(word1_clean, word2_clean) if a != b) + abs(len(word1_clean) - len(word2_clean))
        
        # Return True if the number of different characters is within the allowed maximum
        return diff_chars <= max_diff_chars

    def find_word(self, start: float, end: float, words: List[data.Word], offset: float = 0.3) -> Optional[data.Word]:
        for word in words:
            if abs(word.start - start) <= offset and abs(word.end - end) <= offset:
                return word
        return None

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not dp.data or dp.data.transcribed_segments is None:
            raise Exception("No transcribed words found")
        
        # Collect new words from the transcribed segments
        new_words: List[data.Word] = []
        for segment in dp.data.transcribed_segments:
            if segment.words is None:
                continue
            new_words.extend(segment.words)

        only_words = [word.word for word in new_words]
        print(only_words)
            
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
        time_tolerance = 0.8
        for new_word in list(reversed(new_confirmed)):
            found = False
            for confirmed_word in list(reversed(list(self.confirmed))):
                if abs(confirmed_word.start - new_word.start) <= time_tolerance and abs(confirmed_word.end - new_word.end) <= time_tolerance:
                    if self.is_similar(confirmed_word.word, new_word.word):
                        found = True

                        if confirmed_word.word != new_word.word and confirmed_word.probability - 0.2 < new_word.probability:
                            confirmed_word.word = new_word.word
                            confirmed_word.start = new_word.start
                            confirmed_word.end = new_word.end
                            confirmed_word.probability = new_word.probability

                        break
            if not found:
                self.confirmed.append(new_word)

        # Remove words from confirmed which are not confidant enough < 0.5
        # self.confirmed = [word for word in self.confirmed if word.probability >= 0.6]

        # sort confirmed words by start time
        self.confirmed = sorted(self.confirmed, key=lambda x: x.start)

        # remove words which times are overlapping.
        to_remove_list = []
        for i in range(len(self.confirmed) - 1):
            if self.confirmed[i].end > self.confirmed[i + 1].start:
                if self.is_similar(self.confirmed[i].word, self.confirmed[i + 1].word, 1):
                    to_remove_list.append(i)
                    i = i + 1

        while len(to_remove_list) > 0:
            i = to_remove_list.pop(0)
            self.confirmed.pop(i)
            to_remove_list = [x-1 for x in to_remove_list]


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
