from typing import Optional
import whisper # type: ignore
import torch
import io
import tempfile
from pydub import AudioSegment # type: ignore

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ExecutionModule, ModuleOptions
import logger
import data

log = logger.get_logger()

class Whisper(Module):
    def __init__(self) -> None:
        super().__init__(ModuleOptions(
                                use_mutex=True,
                                timeout=5,
                            ),
                            name="Whisper"
                        )
        self.ram_disk_path = "/mnt/ramdisk" # string: Path to the ramdisk
        self.task = "translate"             # string: transcribe, translate (transcribe or translate it to english)
        self.model = "large"                # string: tiny, base, small, medium, large (Whisper model to use)
        self.models_path = ".models"        # string: Path to the model
        self.english_only = False           # boolean: Only translate to english

        if self.model != "large" and self.english_only:
            self.model = self.model + ".en"

        self._whisper_model: Optional[whisper.Whisper] = None
        
    def init_module(self) -> None:
        log.info(f"Loading model '{self.model}'...")
        self._whisper_model = whisper.load_model(self.model, download_root=self.models_path)
        log.info("Model loaded")
    
    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not self._whisper_model:
            raise Exception("Whisper model not loaded")
        if dp.data:
            log.info(f"Processing {len(dp.data.raw_audio_data)} bytes of audio data")
            if dp.data:
                with tempfile.NamedTemporaryFile(prefix='tmp_audio_', suffix='.wav', dir=self.ram_disk_path, delete=True) as temp_file:
                    # Convert opus to wav
                    opus_data = io.BytesIO(dp.data.raw_audio_data)
                    opus_audio = AudioSegment.from_file(opus_data, format="ogg", frame_rate=48000, channels=2, sample_width=2)
                    opus_audio.export(temp_file.name, format="wav")
                    
                    # Transcribe audio data
                    result = self._whisper_model.transcribe(temp_file.name, fp16=torch.cuda.is_available(), task=self.task)
                    text = result['text'].strip()
                    dp.data.text = text