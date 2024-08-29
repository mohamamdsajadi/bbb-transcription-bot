import torch
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ExecutionModule, ModuleOptions
import logger
import data

log = logger.get_logger()

class Next_Word_Prediction(Module):
    def __init__(self) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=False,
                timeout=5,
            ),
            name="Next_Word_Prediction"
        )
        self.model_name = "meta-llama/Meta-Llama-3.1-8B"
        self.access_token = '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'



    def init_module(self) -> None:
        # Lade Tokenizer und Modell
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=self.access_token)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", load_in_4bit=True, use_auth_token=self.access_token)

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not dp.data or not dp.data.transcribed_text:
            raise ValueError("DataPackage is empty")
        
        # dp.data.transcribed_text["confirmed_words"] # List of confirmed words
        # dp.data.transcribed_text["unconfirmed_words"] # List of next word predictions
        confirmed_words: str = ' '.join(word for word in dp.data.transcribed_text["confirmed_words"])
        unconfirmed_words: str = ' '.join(word for word in dp.data.transcribed_text["unconfirmed_words"])
        input_text = confirmed_words + ' ' + unconfirmed_words
        encoded_text = self.tokenizer(input_text, return_tensors="pt").to('cuda')

        input_ids = encoded_text['input_ids']

        probabilities = []

        while True:
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model(input_ids)
            
            next_token_logits = outputs.logits[0, -1, :]
            
            # Wahrscheinlichkeiten aus Logits berechnen
            next_token_probs = torch.softmax(next_token_logits, -1)
            
            # Nächstes Token mit höchster Wahrscheinlichkeit auswählen
            next_token_id = torch.argmax(next_token_probs).unsqueeze(0)
            next_token_prob = next_token_probs[next_token_id].item()
            
            # Wahrscheinlichkeiten und das nächste Token speichern
            probabilities.append((self.tokenizer.decode(next_token_id), next_token_prob))
            
            # Update der Input-IDs mit dem neu generierten Token
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

            # Abbruchbedingung: Wahrscheinlichkeit unter 30%
            if next_token_prob < 0.8:
                break

        transcribed_text = []
        for word, prob in probabilities:
            # print(f"Wort: {word}, Wahrscheinlichkeit: {prob:.4f}")
            transcribed_text.append(word)

        dp.data.transcribed_text["next_word"] = []
        for word, prob in probabilities:
            # print(f"Wort: {word}, Wahrscheinlichkeit: {prob:.4f}")
            dp.data.transcribed_text["next_word"].append(word)