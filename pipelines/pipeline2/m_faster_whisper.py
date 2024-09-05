# m_faster_whisper.py
from typing import List, Tuple

import numpy as np
import ctranslate2  # type: ignore
import faster_whisper  # type: ignore
from whisperx.audio import log_mel_spectrogram  # type: ignore
import torch

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ModuleOptions
import logger
import data


log = logger.get_logger()

class WhisperModel(faster_whisper.WhisperModel):
    '''
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    '''

    def generate_segment_batched(
        self,
        features: np.ndarray,
        tokenizer: faster_whisper.tokenizer.Tokenizer,
        options: faster_whisper.transcribe.TranscriptionOptions,
        encoder_output: ctranslate2.StorageView = None
    ) -> List[str]:
        batch_size: int = features.shape[0]
        all_tokens: List[int] = []
        prompt_reset_since: int = 0
        if options.initial_prompt is not None:
            initial_prompt: str = " " + options.initial_prompt.strip()
            initial_prompt_tokens: List[int] = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens: List[int] = all_tokens[prompt_reset_since:]
        prompt: List[int] = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        encoder_output = self.encode(features)

        result: List[ctranslate2.TranslationResult] = self.model.generate(
            encoder_output,
            [prompt] * batch_size,
            beam_size=options.beam_size,
            patience=options.patience,
            length_penalty=options.length_penalty,
            max_length=self.max_length,
            suppress_blank=options.suppress_blank,
            suppress_tokens=options.suppress_tokens,
        )

        tokens_batch: List[List[int]] = [x.sequences_ids[0] for x in result]

        def decode_batch(tokens: List[List[int]]) -> List[str]:
            res: List[List[int]] = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            return tokenizer.tokenizer.decode_batch(res)

        text: List[str] = decode_batch(tokens_batch)

        return text

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        to_cpu: bool = self.model.device == "cuda" and len(self.model.device_index) > 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        new_features: ctranslate2.StorageView = faster_whisper.transcribe.get_ctranslate2_storage(features)

        return self.model.encode(new_features, to_cpu=to_cpu)
    
    def detect_language(self, audio: np.ndarray, sample_rate: int, chunk_length: int) -> Tuple[str, float]:
        n_samples: int = chunk_length * sample_rate  # 480000 samples in a 30-second chunk
        
        model_n_mels: int = 128
        
        segment: np.ndarray = log_mel_spectrogram(
            audio[:n_samples],
            n_mels=model_n_mels,
            padding=0 if audio.shape[0] >= n_samples else n_samples - audio.shape[0]
        )
        
        if not isinstance(segment, np.ndarray):
            segment = np.array(segment)

        segment = np.expand_dims(segment, axis=0)

        segment_storage_view: ctranslate2.StorageView = ctranslate2.StorageView.from_array(segment.astype(np.float32))

        encoder_output: ctranslate2.StorageView = self.model.encode(segment_storage_view, to_cpu=False)
        
        results: List[List[Tuple[str, float]]] = self.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language: str = language_token[2:-2]
        
        print(f"Detected language: {language} ({language_probability:.2f})")
        return language, language_probability

class Faster_Whisper_transcribe(Module):
    def __init__(self) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=True,
                timeout=5,
            ),
            name="Faster_Whisper_transcribe"
        )
        self.model_size: str = "large-v3"
        self.compute_type: str = "float16"
        self.batch_size: int = 32
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_chunk_to_detect_language: int = 30
        self.task: str = "transcribe"
        self.asr_options: faster_whisper.transcribe.TranscriptionOptions = faster_whisper.transcribe.TranscriptionOptions(**{
            "beam_size": 5,
            "best_of": 1,
            "patience": 1.0,
            "length_penalty": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": False,
            "prompt_reset_on_temperature": 0.5,
            "initial_prompt": None,
            "prefix": None,
            "suppress_blank": True,
            "suppress_tokens": None,
            "without_timestamps": True,
            "max_initial_timestamp": 0.0,
            "word_timestamps": False,
            "prepend_punctuations": "\"'“¿([{-",
            "append_punctuations": "\"'.。,，!！?？:：”)]}、",
            "max_new_tokens": 50,
            "clip_timestamps": 60,
            "hallucination_silence_threshold": 0.5,
            "hotwords": None,
        })

    def init_module(self) -> None:
        log.info(f"Loading model whisper:'{self.model_size}'...")
        self.model: WhisperModel = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        log.info("Model loaded")

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not self.model:
            raise Exception("Whisper model not loaded")
        if not dp.data:
            raise Exception("No data found")
        if dp.data.audio_data is None:
            raise Exception("No audio data found")
        if not dp.data.vad_result:
            raise Exception("No audio data from VAD found")
        if not dp.data.audio_data_sample_rate:
            raise Exception("No sample rate found")
        
        # Detect language with confidence threshold
        language, confidence = self.model.detect_language(dp.data.audio_data, dp.data.audio_data_sample_rate, self.audio_chunk_to_detect_language)
        dp.data.language = (language, confidence)
        
        # Initialize tokenizer for transcription 
        tokenizer: faster_whisper.tokenizer.Tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer, self.model.model.is_multilingual, task=self.task, language=language)
        
        # Transcribe VAD-based voice segments in batches
        batched_segments: List[np.ndarray] = []
        batched_times: List[Tuple[float, float]] = []  # To keep track of start and end times for each segment
        max_length: int = 0
        
        segment_audio = dp.data.audio_data
        merged_segments = dp.data.vad_result
        sample_rate = dp.data.audio_data_sample_rate
        
        result: List[data.TextSegment] = []
        
        # Calculate the maximum length of segments
        for idx, segment in enumerate(merged_segments):
            start_time: float = segment['start']
            end_time: float = segment['end']

            # Extract only the relevant segment based on start and end times
            start_sample: int = int(start_time * sample_rate)
            end_sample: int = int(end_time * sample_rate)
            audio_segment: np.ndarray = segment_audio[start_sample:end_sample]

            # Calculate the maximum length of the segments
            max_length = max(max_length, len(audio_segment))
            batched_segments.append(audio_segment)
            batched_times.append((start_time, end_time))

            # Transcribe the segments in batches
            if len(batched_segments) == self.batch_size or idx == len(merged_segments) - 1:
                # Convert segments to mel-spectrograms and pad all segments to the same length
                mel_segments: List[np.ndarray] = [log_mel_spectrogram(seg, padding=0, n_mels=128) for seg in batched_segments]

                # Find the maximum number of frames (time steps) in the mel-spectrograms
                max_frames: int = max(mel.shape[1] for mel in mel_segments)

                # Pad each mel-spectrogram to the maximum frame length
                padded_mel_segments: List[np.ndarray] = [np.pad(mel, ((0, 0), (0, max_frames - mel.shape[1])), mode='constant') for mel in mel_segments]

                # Convert padded mel segments to a numpy array
                mel_segments_array: np.ndarray = np.array(padded_mel_segments, dtype=np.float32)  # Use float32 for compatibility

                # Use the generate_segment_batched method for transcription
                transcriptions: List[str] = self.model.generate_segment_batched(
                    features=mel_segments_array,
                    tokenizer=tokenizer,  # Use the correct tokenizer
                    options=self.asr_options,
                )

                # Output the results
                for i, text in enumerate(transcriptions):
                    start_time_str = batched_times[i][0]
                    end_time_str = batched_times[i][1]
                    
                    # Ensure creation of the correct 'TextSegment' object
                    text_segment = data.TextSegment(
                        text=text,
                        start=start_time_str,
                        end=end_time_str
                    )

                    # Append to the list 'result' which expects 'TextSegment' objects
                    result.append(text_segment)
                    
                    # print(f"Batch {i + 1} [{start_time_str}->{end_time_str}]: {text}")

                # Clear batched_segments and batched_times for the next batch
                batched_segments = []
                batched_times = []
                max_length = 0  # Reset the maximum length for the next batch
                
        dp.data.transcribed_segments = result

        # The loop that concatenates all text segments into a single string
        complete_text = ""
        for seg in result:
            complete_text += seg.text + " "
            
        dp.data.transcribed_text = complete_text

        # Split the complete text into words
        words = complete_text.split()
        dp.data.transcribed_words = words