import hashlib
from typing import Any, Callable, Dict, List, Optional, Text, Union

import numpy as np
from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule, Status
from stream_pipeline.module_classes import Module, ExecutionModule, ModuleOptions
import data

import os
import urllib
from tqdm import tqdm # type: ignore
import torch
from whisperx.vad import merge_chunks  # type: ignore

from pyannote.audio import Model # type: ignore
from pyannote.audio.pipelines import VoiceActivityDetection # type: ignore
from pyannote.audio.pipelines.utils import PipelineModel # type: ignore
from pyannote.core import Annotation, Segment, SlidingWindowFeature # type: ignore

import logger
log = logger.get_logger()

class VoiceActivitySegmentation(VoiceActivityDetection):
    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        fscore: bool = False,
        use_auth_token: Union[Text, None] = None,
        **inference_kwargs: Any,
    ) -> None:
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
        if not dp.data.audio_data_sample_rate:
            raise Exception("No sample rate found")
        
        # Perform voice activity detection
        vad_result: Annotation = self.model.apply(dp.data.audio_data, sr=dp.data.audio_data_sample_rate)
        
        # Merge VAD segments if necessary
        merged_segments: List[Dict[str, float]] = merge_chunks(vad_result, chunk_size=self.chunk_size)
        
        if len(merged_segments) == 0:
            dpm.message = "No voice detected"
            dpm.status = Status.EXIT
        
        dp.data.vad_result = merged_segments
        
    

    def load_vad_model(self, device: str, vad_onset: float=0.500, vad_offset: float=0.363, use_auth_token: Union[Text, None]=None, model_fp: Union[Text, None]=None) -> VoiceActivitySegmentation:
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