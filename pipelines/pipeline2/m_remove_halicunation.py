# m_local_agreement.py
import string
from typing import List

from difflib import SequenceMatcher

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ModuleOptions

import logger
import data

log = logger.get_logger()

class Remove_Hallucination(Module):
    def __init__(self) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=False,
                timeout=5,
            ),
            name="Remove_Hallucination"
        )
        
        self.max_distance = 10
        self.similarity_threshold = 0.95

    def init_module(self) -> None:
        pass

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not dp.data or dp.data.transcribed_words is None:
            raise Exception("No transcribed words found")
        
        words = dp.data.transcribed_words
        new_words = self.remove_similar_phrases(words, self.max_distance, self.similarity_threshold)
        
        dp.data.cleaned_words = new_words
        

    def preprocess_word(self, word):
        """
        Preprocesses a word by converting it to lowercase and removing punctuation.
        
        Args:
            word (str): The word to preprocess.
            
        Returns:
            str: The preprocessed word.
        """
        return word.lower().translate(str.maketrans('', '', string.punctuation))

    def is_similar(self, seq1, seq2, threshold=0.9):
        """
        Determines if two sequences are similar based on a given threshold.
        
        Args:
            seq1 (list): The first sequence of words.
            seq2 (list): The second sequence of words.
            threshold (float): The similarity threshold (between 0 and 1).
            
        Returns:
            bool: True if the sequences are similar, False otherwise.
        """
        # Preprocess sequences for similarity comparison
        preprocessed_seq1 = [self.preprocess_word(word) for word in seq1]
        preprocessed_seq2 = [self.preprocess_word(word) for word in seq2]
        
        ratio = SequenceMatcher(None, preprocessed_seq1, preprocessed_seq2).ratio()
        return ratio >= threshold

    def remove_similar_phrases(self, words, max_distance, similarity_threshold=0.9):
        """
        Removes repeated similar phrases from a list of words based on a similarity threshold.
        
        Args:
            words (list): List of words.
            max_distance (int): Maximum distance to look ahead for repeated sequences.
            similarity_threshold (float): The similarity threshold for comparison.
            
        Returns:
            list: List of words with similar repeated sequences removed.
        """
        # Initialize a result list
        result = []
        i = 0
        n = len(words)
        
        while i < n:
            # Determine the max phrase length to check for repetition
            max_phrase_length = min(n - i, max_distance)
            
            # Check for repeated similar sequences of words
            found_repetition = False
            for length in range(1, max_phrase_length + 1):
                phrase = words[i:i + length]
                next_start = i + length
                
                # Compare the current phrase with the next phrase of the same length
                while next_start + length <= n:
                    next_phrase = words[next_start:next_start + length]
                    if self.is_similar(phrase, next_phrase, similarity_threshold):
                        # If a similar phrase is found, skip all further occurrences
                        found_repetition = True
                        next_start += length
                    else:
                        break
                
                # If repetition is found, only keep the first occurrence and skip the rest
                if found_repetition:
                    result.extend(phrase)
                    i = next_start
                    break
            
            # If no repetition is found, add the word to the result
            if not found_repetition:
                result.append(words[i])
                i += 1
        
        return result
        
    