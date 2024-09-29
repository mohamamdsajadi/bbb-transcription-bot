# m_local_agreement.py
from typing import List, Dict, Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer # type: ignore
import torch

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ModuleOptions

import logger
import data

log = logger.get_logger()

class Translate(Module):
    def __init__(self) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=False,
                timeout=5,
            ),
            name="Translate"
        )
        self.model_name: str = "facebook/nllb-200-distilled-600M"
        # List of target languages
        self.target_languages: List[str] = [
            "deu_Latn",  # German
            "fra_Latn",  # French
            "spa_Latn",  # Spanish
        ]

        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def init_module(self) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)

    def translate_to_language(self, source_lang: str, target_lang: str, text: str) -> str:
        if self.model is None:
            raise Exception("Model not initialized")

        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=source_lang)

        inputs = tokenizer(text, return_tensors="pt").to(self.device)

        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang), max_length=30
        )
        translated_text: str = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not dp.data or dp.data.confirmed_words is None:
            raise Exception("No transcribed words found")
        if dp.data.language is None:
            raise Exception("No language detected")
        
        source_lang: str = dp.data.language[0]
        words: List[str] = dp.data.confirmed_words
        text: str = " ".join(words)

        full_translations: Dict[str, str] = {}

        for target_lang in self.target_languages:
            translation: str = self.translate_to_language(source_lang, target_lang, text)
            full_translations[target_lang] = translation

        dp.data.translations = full_translations
