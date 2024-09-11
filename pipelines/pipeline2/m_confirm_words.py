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
        self.max_confirmed_words = 50
        self.confirmed: List[data.Word] = []  # Buffer to store committed words

        self.confirmed_end_time: float = 0.0
        
        self.confirm_if_older_then: float = 2.0 # Confirm words if they are older than this value in seconds

    def init_module(self) -> None:
        pass

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

        # Ensure that the number of confirmed words does not exceed the max_confirmed_words limit
        if len(self.confirmed) > self.max_confirmed_words:
            self.confirmed = self.confirmed[-self.max_confirmed_words:]

        if len(self.confirmed) > 0:
            self.confirmed_end_time = self.confirmed[-1].end
        
        # Update data package confirmed and unconfirmed words
        dp.data.confirmed_words = self.confirmed.copy()
        dp.data.unconfirmed_words = new_unconfirmed
