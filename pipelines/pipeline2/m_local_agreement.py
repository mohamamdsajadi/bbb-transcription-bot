# m_local_agreement.py
from typing import List

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ModuleOptions

import logger
import data

log = logger.get_logger()

class Local_Agreement(Module):
    def __init__(self) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=False,
                timeout=5,
            ),
            name="Local_Agreement"
        )
        self.max_confirmed_words = 50

        self.unconfirmed: List[str] = []  # To store unconfirmed words
        self.confirmed: List[str] = []    # To store confirmed words

    def init_module(self) -> None:
        pass

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not dp.data or dp.data.transcribed_words is None:
            raise Exception("No transcribed words found")

        new_words = dp.data.transcribed_words
        
        def create_confirmed_and_unconfirmed_lists(new_words: List[str]) -> None:
            # Find the longest matching prefix between current_words and _unconfirmed_words
            for a, con_word in reversed(list(enumerate(self.unconfirmed))):
                for b, new_word in reversed(list(enumerate(new_words))):
                    if con_word == new_word:
                        # Now we maybe know where the last unconfirmed word is in the new words list
                        # We can now start from here and go backwards to find the common prefix
                        # find the common prefix
                        common_prefix = 0
                        temp_un_word_list = list(reversed(self.unconfirmed[:a + 1]))
                        temp_new_word_list = list(reversed(new_words[:b + 1]))
                        for i in range(min(len(temp_un_word_list), len(temp_new_word_list))):
                            if temp_un_word_list[i] == temp_new_word_list[i]:
                                common_prefix += 1
                            else:
                                break
                            
                        
                        # common_prefix has to be exactly the length of the unconfirmed word - processed unconfirmed
                        processed_unconfirmed = len(temp_un_word_list)
                        if common_prefix == processed_unconfirmed:
                            # We can now confirm all unconfirmed to a
                            self.confirmed.extend(self.unconfirmed[:a + 1])
                            self.unconfirmed = self.unconfirmed[a + 1:] + new_words[b + 1:]
                            return
                        else:
                            break
                
                # If we reach here, it means this unconfirmed word doesnt exist in the new words list
                # or the previous unconfirmed word changed. Remove it
                self.unconfirmed = self.unconfirmed[:a]
            
                        
            # Find the longest matching prefix between current_words and confirmed_words
            for a, con_word in reversed(list(enumerate(self.confirmed))):
                for b, new_word in reversed(list(enumerate(new_words))):
                    if con_word == new_word:
                        # Now we maybe know where the last unconfirmed word is in the new words list
                        # We can now start from here and go backwards to find the common prefix
                        # find the common prefix
                        common_prefix = 0
                        temp_un_word_list = list(reversed(self.confirmed[:a + 1]))
                        temp_new_word_list = list(reversed(new_words[:b + 1]))
                        for i in range(min(len(temp_un_word_list), len(temp_new_word_list))):
                            if temp_un_word_list[i] == temp_new_word_list[i]:
                                common_prefix += 1
                            else:
                                break
                
                        if common_prefix > 2:
                            self.unconfirmed = new_words[b + 1:]
                            return
                        
            
            self.unconfirmed = new_words
        
        create_confirmed_and_unconfirmed_lists(new_words)
        
        if len(self.confirmed) > self.max_confirmed_words:
            self.confirmed = self.confirmed[-self.max_confirmed_words:]
        
        dp.data.confirmed_words = self.confirmed.copy()
        dp.data.unconfirmed_words = self.unconfirmed.copy()