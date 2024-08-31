import hashlib
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union
import numpy as np
import urllib

from extract_ogg import OggSFrame, calculate_frame_duration, get_header_frames

from whisperx.vad import merge_chunks  # type: ignore
from pyannote.audio import Model # type: ignore
from tqdm import tqdm # type: ignore
from pyannote.audio.pipelines import VoiceActivityDetection # type: ignore
from pyannote.audio.pipelines.utils import PipelineModel # type: ignore
from pyannote.core import Annotation, Segment, SlidingWindowFeature # type: ignore

from whisperx.audio import log_mel_spectrogram  # type: ignore
import ctranslate2  # type: ignore
import faster_whisper  # type: ignore

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule, Status
from stream_pipeline.module_classes import Module, ExecutionModule, ModuleOptions
import torch
import logger
import data

log = logger.get_logger()




class CreateNsAudioPackage(ExecutionModule):
    def __init__(self) -> None:
        super().__init__(ModuleOptions(
                                use_mutex=False,
                                timeout=5,
                            ),
                            name="Create10sAudioPackage"
                        )
        self.audio_data_buffer: List[OggSFrame] = []
        self.sample_rate: int = 48000
        self.last_n_seconds: int = 10
        self.min_n_seconds: int = 1
        self.current_audio_buffer_seconds: float = 0

    

        self.header_buffer: bytes = b''
        self.header_frames: Optional[List[OggSFrame]] = None

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if dp.data:
            frame = OggSFrame(dp.data.raw_audio_data)

            if not self.header_frames:
                self.header_buffer += frame.raw_data
                id_header_frame, comment_header_frames = get_header_frames(self.header_buffer)

                if id_header_frame and comment_header_frames:
                    self.header_frames = []
                    self.header_frames.append(id_header_frame)
                    self.header_frames.extend(comment_header_frames)
                else:
                    dpm.message = "Could not find the header frames"
                    dpm.status = Status.EXIT
                    return

            

            last_frame: Optional[OggSFrame] = self.audio_data_buffer[-1] if len(self.audio_data_buffer) > 0 else None

            current_granule_position: int = frame.header['granule_position']
            previous_granule_position: int = last_frame.header['granule_position'] if last_frame else 0

            frame_duration: float = calculate_frame_duration(current_granule_position, previous_granule_position, self.sample_rate)
            previous_granule_position = current_granule_position


            self.audio_data_buffer.append(frame)
            self.current_audio_buffer_seconds += frame_duration

            # Every second, process the last n seconds of audio
            if frame_duration > 0.0 and self.current_audio_buffer_seconds >= self.min_n_seconds:
                if self.current_audio_buffer_seconds >= self.last_n_seconds:
                    # pop audio last frame from buffer
                    pop_frame = self.audio_data_buffer.pop(0)
                    pop_frame_granule_position = pop_frame.header['granule_position']
                    next_frame_granule_position = self.audio_data_buffer[0].header['granule_position'] if len(self.audio_data_buffer) > 0 else pop_frame_granule_position
                    pop_frame_duration = calculate_frame_duration(next_frame_granule_position, pop_frame_granule_position, self.sample_rate)
                    self.current_audio_buffer_seconds -= pop_frame_duration

                # Combine the audio buffer into a single audio package
                n_seconds_of_audio: bytes = self.header_buffer + b''.join([frame.raw_data for frame in self.audio_data_buffer])
                dp.data.raw_audio_data = n_seconds_of_audio
            else:
                dpm.status = Status.EXIT
                dpm.message = "Not enough audio data to create a package"






class Load_audio(Module):
    def __init__(self) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=False,
                timeout=5,
            ),
            name="Load_audio"
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
        if not dp.data:
            raise Exception("No data found")
        if not dp.data.raw_audio_data:
            raise Exception("No audio data found")
        if not dp.data.sample_rate:
            raise Exception("No sample rate found")
        
        audio_data = self.load_audio_from_binary(dp.data.raw_audio_data, dp.data.sample_rate)
        dp.data.audio_data = audio_data






class VoiceActivitySegmentation(VoiceActivityDetection):
    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        fscore: bool = False,
        use_auth_token: Union[Text, None] = None,
        **inference_kwargs,
    ):
        super().__init__(segmentation=segmentation, fscore=fscore, use_auth_token=use_auth_token, **inference_kwargs)

    def apply(self, audio_waveform: np.ndarray, sr: int = 16000, hook: Optional[Callable] = None) -> Annotation:
        """Apply voice activity detection to a NumPy array waveform.

        Parameters
        ----------
        audio_waveform : np.ndarray
            NumPy array containing the audio waveform.
        sr : int
            Sample rate of the audio waveform.
        hook : callable, optional
            Hook called after each major step of the pipeline with the following
            signature: hook("step_name", step_artefact, file=file)

        Returns
        -------
        speech : Annotation
            Speech regions.
        """

        # If a hook is provided, set it up (e.g., for debugging purposes)
        if hook is not None:
            hook("start", None, file=None)

        # Convert numpy array to PyTorch tensor
        waveform_tensor = torch.tensor(audio_waveform[None, :], dtype=torch.float32)  # Add batch dimension

        # Prepare the input as a dictionary with waveform (tensor) and sample rate
        input_dict = {
            "waveform": waveform_tensor,  # Use the PyTorch tensor here
            "sample_rate": sr
        }

        # Process the waveform using the segmentation model
        segmentations: SlidingWindowFeature = self._segmentation(input_dict)

        # Call hook after segmentation step if provided
        if hook is not None:
            hook("segmentation", segmentations, file=None)

        return segmentations



class VAD(Module):
    def __init__(self) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=True,
                timeout=5,
            ),
            name="VAD"
        )
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[VoiceActivitySegmentation] = None
        self.vad_segmentation_url = "https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin"

        self.vad_onset = 0.500
        self.vad_offset = 0.363
        self.use_auth_token=None
        self.model_fp=None
        self.chunk_size = 30.0

    def init_module(self) -> None:
        log.info(f"Loading model vad...")
        self.model = self.load_vad_model(device=self.device, vad_onset=self.vad_onset, vad_offset=self.vad_offset, use_auth_token=self.use_auth_token, model_fp=self.model_fp)
        log.info("Model loaded")

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not self.model:
            raise Exception("Whisper model not loaded")
        if not dp.data:
            raise Exception("No data found")
        if dp.data.audio_data is None:
            raise Exception("No audio data found")
        if not dp.data.sample_rate:
            raise Exception("No sample rate found")
        
        # Perform voice activity detection
        vad_result: Annotation = self.model.apply(dp.data.audio_data, sr=dp.data.sample_rate)
        
        # Merge VAD segments if necessary
        merged_segments: List[Dict[str, float]] = merge_chunks(vad_result, chunk_size=self.chunk_size)
        
        dp.data.vad_result = merged_segments
        
    

    def load_vad_model(self, device, vad_onset=0.500, vad_offset=0.363, use_auth_token=None, model_fp=None):
        model_dir = torch.hub._get_torch_home()
        os.makedirs(model_dir, exist_ok = True)
        if model_fp is None:
            model_fp = os.path.join(model_dir, "whisperx-vad-segmentation.bin")
        if os.path.exists(model_fp) and not os.path.isfile(model_fp):
            raise RuntimeError(f"{model_fp} exists and is not a regular file")

        if not os.path.isfile(model_fp):
            with urllib.request.urlopen(self.vad_segmentation_url) as source, open(model_fp, "wb") as output:
                with tqdm(
                    total=int(source.info().get("Content-Length")),
                    ncols=80,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as loop:
                    while True:
                        buffer = source.read(8192)
                        if not buffer:
                            break

                        output.write(buffer)
                        loop.update(len(buffer))

        model_bytes = open(model_fp, "rb").read()
        if hashlib.sha256(model_bytes).hexdigest() != self.vad_segmentation_url.split('/')[-2]:
            raise RuntimeError(
                "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
            )

        vad_model = Model.from_pretrained(model_fp, use_auth_token=use_auth_token)
        hyperparameters = {"onset": vad_onset, 
                        "offset": vad_offset,
                        "min_duration_on": 0.1,
                        "min_duration_off": 0.1}
        vad_pipeline = VoiceActivitySegmentation(segmentation=vad_model, device=torch.device(device))
        vad_pipeline.instantiate(hyperparameters)

        return vad_pipeline










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
        if not dp.data.sample_rate:
            raise Exception("No sample rate found")
        
        # Detect language with confidence threshold
        language, confidence = self.model.detect_language(dp.data.audio_data, dp.data.sample_rate, self.audio_chunk_to_detect_language)
        dp.data.language = (language, confidence)
        
        # Initialize tokenizer for transcription 
        tokenizer: faster_whisper.tokenizer.Tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer, self.model.model.is_multilingual, task=self.task, language=language)
        
        # Transcribe VAD-based voice segments in batches
        batched_segments: List[np.ndarray] = []
        batched_times: List[Tuple[float, float]] = []  # To keep track of start and end times for each segment
        max_length: int = 0
        
        segment_audio = dp.data.audio_data
        merged_segments = dp.data.vad_result
        sample_rate = dp.data.sample_rate
        
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
        
        dp.data.confirmed_words = self.confirmed
        dp.data.unconfirmed_words = self.unconfirmed
