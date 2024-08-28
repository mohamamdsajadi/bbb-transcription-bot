# asr_whisperx.py
from typing import List, Optional, Union, Dict, Any
import whisperx  # type: ignore
from pydub import AudioSegment  # type: ignore

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ExecutionModule, ModuleOptions
import logger
import data

log = logger.get_logger()

import subprocess
import numpy as np


class WhisperX_load_audio(Module):
    def __init__(self) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=False,
                timeout=5,
            ),
            name="WhisperX_load_audio"
        )

    def init_module(self) -> None:
        pass

    def load_audio_from_binary(self, data: bytes, sr: int = 16000) -> np.ndarray:
        """
        Process binary audio data (e.g., OGG Opus) and convert it to a mono waveform, resampling as necessary.

        Parameters
        ----------
        data: bytes
            The binary audio data to process.

        sr: int
            The sample rate to resample the audio if necessary.

        Returns
        -------
        np.ndarray
            A NumPy array containing the audio waveform, in float32 dtype.
        """
        try:
            # Set up the ffmpeg command to read from a pipe
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-threads",
                "0",
                "-i", "pipe:0",  # Use the pipe:0 to read from stdin
                "-f", "s16le",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                "-ar", str(sr),
                "-",
            ]

            # Run the ffmpeg process, feeding it the binary data through stdin
            process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, err = process.communicate(input=data)

            if process.returncode != 0:
                raise RuntimeError(f"Failed to load audio: {err.decode()}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        # Convert the raw audio data to a NumPy array and normalize it
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def execute(
        self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule
    ) -> None:
        if dp.data:
            audio_data = self.load_audio_from_binary(dp.data.raw_audio_data)
            dp.data.audio_data = audio_data


class WhisperX_transcribe(Module):
    def __init__(self) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=True,
                timeout=5,
            ),
            name="WhisperX_transcribe"
        )
        self.model: Optional[Any] = None
        self.model_name: str = "large-v3"
        self.device: str = "cuda"
        self.audio_file: str = "audio/audio-de.mp3"
        self.batch_size: int = 32  # reduce if low on GPU mem
        self.compute_type: str = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)

    def init_module(self) -> None:
        log.info(f"Loading model '{self.model_name}'...")
        self.model = whisperx.load_model(self.model_name, self.device, compute_type=self.compute_type)
        log.info("Model loaded")

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not self.model:
            raise Exception("Whisper model not loaded")
        if not dp.data:
            raise Exception("No audio data found")
            
        result: Dict[str, Any] = self.model.transcribe(dp.data.audio_data, batch_size=self.batch_size)
        dp.data.transcribed_text = result


class WhisperX_align(Module):
    def __init__(self) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=True,
                timeout=5,
            ),
            name="WhisperX"
        )
        self.device: str = "cuda"
        self.model: Optional[Any] = None
        self.metadata: Optional[Dict[str, Any]] = None

    def init_module(self) -> None:
        log.info("Loading align model...")
        self.model, self.metadata = whisperx.load_align_model(language_code="en", device=self.device)
        log.info("Align model loaded")

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not self.model:
            raise Exception("Whisper align model not loaded")
        if not dp.data:
            raise Exception("No audio data found")
            
        metadata = self.metadata.copy() if self.metadata else {}
        metadata["language"] = dp.data.transcribed_text["language"] if dp.data.transcribed_text else "en"
        segments = dp.data.transcribed_text["segments"] if dp.data.transcribed_text else []
        result: Dict[str, Any] = whisperx.align(
            segments, self.model, metadata, dp.data.audio_data, self.device, return_char_alignments=False
        )
        dp.data.transcribed_text = result


class Clean_Whisper_data(Module):
    def __init__(self) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=False,
                timeout=5,
            ),
            name="Clean_Whisper_data"
        )

    def init_module(self) -> None:
        pass

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not dp.data or not dp.data.transcribed_text:
            raise Exception("No transcribed text data found")
        
        complete_text = ""
        
        # Add all segments together
        for segment in dp.data.transcribed_text["segments"]:
            complete_text += str(segment["text"]) + " "
            
        
        dp.data.transcribed_text["text"] = complete_text
            
        # Split the text into words
        dp.data.transcribed_text["words"] = complete_text.split()
        
        


class Local_Agreement(Module):
    def __init__(self) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=False,
                timeout=5,
            ),
            name="Local_Agreement"
        )

        self.unconfirmed: List[str] = []  # To store unconfirmed words
        self.confirmed: List[str] = []    # To store confirmed words

    def init_module(self) -> None:
        pass

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not dp.data or not dp.data.transcribed_text:
            raise Exception("No transcribed text data found")

        new_words = dp.data.transcribed_text["words"]
        
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
        
        dp.data.transcribed_text["confirmed_words"] = self.confirmed
        dp.data.transcribed_text["unconfirmed_words"] = self.unconfirmed
