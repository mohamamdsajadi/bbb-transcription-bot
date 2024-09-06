import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="ron_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

article = "Şeful ONU spune că nu există o soluţie militară în Siria"

# Tokenize input and move to GPU if available
inputs = tokenizer(article, return_tensors="pt").to(device)

# List of target languages
target_languages = [
    "deu_Latn",  # German
    "fra_Latn",  # French
    "spa_Latn",  # Spanish
    "ita_Latn",  # Italian
    "por_Latn",  # Portuguese
    "rus_Cyrl",  # Russian
    "zho_Hans",  # Chinese (Simplified)
    "jpn_Jpan",  # Japanese
    "kor_Hang",  # Korean
    "ara_Arab",  # Arabic
    "hin_Deva",  # Hindi
    "tur_Latn",  # Turkish
    "nld_Latn",  # Dutch
    "swe_Latn",  # Swedish
    "nor_Latn",  # Norwegian
    "dan_Latn",  # Danish
    "fin_Latn",  # Finnish
    "ell_Grek",  # Greek
    "heb_Hebr",  # Hebrew
    "vie_Latn",  # Vietnamese
    "tha_Thai",  # Thai
    "pol_Latn",  # Polish
    "ukr_Cyrl",  # Ukrainian
    "ron_Latn",  # Romanian
    "hun_Latn",  # Hungarian
    "ces_Latn",  # Czech
    "bul_Cyrl",  # Bulgarian
    "hrv_Latn",  # Croatian
    "srp_Cyrl",  # Serbian
    "slv_Latn",  # Slovenian
    "lit_Latn",  # Lithuanian
    "lav_Latn",  # Latvian
    "est_Latn",  # Estonian
    "ben_Beng",  # Bengali
    "tam_Taml",  # Tamil
    "tel_Telu",  # Telugu
    "urd_Arab",  # Urdu
    "mlt_Latn",  # Maltese
    "isl_Latn",  # Icelandic
    "msa_Latn",  # Malay
    "ind_Latn",  # Indonesian
    "cat_Latn",  # Catalan
    "eus_Latn",  # Basque
    "glg_Latn",  # Galician
    "hye_Armn",  # Armenian
    "kaz_Cyrl",  # Kazakh
    "uzb_Latn",  # Uzbek
    "aze_Latn",  # Azerbaijani
    "afr_Latn",  # Afrikaans
    "swa_Latn",  # Swahili
    "nob_Latn"   # Norwegian (Bokmål)
]


# Function to translate to a specific language
def translate_to_language(target_lang):
    start = time.time()
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang), max_length=30
    )
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    end = time.time()
    return target_lang, translated_text, end - start

# Parallel translation
start_all = time.time()
with ThreadPoolExecutor(max_workers=len(target_languages)) as executor:
    futures = [executor.submit(translate_to_language, lang) for lang in target_languages]

    # Print the results as they are completed
    for future in as_completed(futures):
        lang, text, duration = future.result()
        print(f"Translation to {lang}: {text} (Time taken: {duration:.2f} seconds)")

end_all = time.time()
print(f"Total time taken for {len(target_languages)} languages to translate: {end_all - start_all:.2f} seconds")
