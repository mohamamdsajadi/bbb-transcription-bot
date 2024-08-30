from whisperx.vad import load_vad_model, merge_chunks
from faster_whisper import WhisperModel
from pyannote.audio.core.io import AudioFile
import numpy as np
import subprocess
import torch

# Definition der Funktion load_audio
SAMPLE_RATE = 16000  # Definiere SAMPLE_RATE global oder importiere es, wenn es bereits definiert ist

def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary.

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

# Schritt 1: VAD-Modell laden und konfigurieren
device = "cuda"  # oder "cpu" je nach Verfügbarkeit
vad_model = load_vad_model(device=device)

# Eingabe-Audiodatei
audio_file_path = "audio/audio.mp3"
audio_file = {"uri": "audio_sample", "audio": audio_file_path}

# Schritt 2: Sprachaktivitätserkennung durchführen
vad_result = vad_model.apply(audio_file)
# Zusammenführung von VAD-Segmenten, falls notwendig
merged_segments = merge_chunks(vad_result, chunk_size=30.0)  # Optional: Passen Sie die chunk_size an

# Schritt 3: Whisper-Modell laden und konfigurieren
model_size = "large-v3"
model = WhisperModel(model_size, device=device, compute_type="float16")

# Extrahiere das Audio-Segment basierend auf den VAD-Ergebnissen
segment_audio = load_audio(audio_file_path)  # Laden Sie die gesamte Audiodatei

# Schritt 4: Transkription der VAD-basierten Sprachsegmente
for idx, segment in enumerate(merged_segments):
    start_time = segment['start']
    end_time = segment['end']



    # Extrahiere nur das relevante Segment basierend auf den Start- und Endzeiten
    start_sample = int(start_time * SAMPLE_RATE)
    end_sample = int(end_time * SAMPLE_RATE)
    audio_segment = segment_audio[start_sample:end_sample]

    # Transkribiere nur das aktuelle Segment
    segments, info = model.transcribe(
        audio_segment,
        beam_size=5,
        word_timestamps=False,
        condition_on_previous_text=False,
    )

    # Ausgabe der Ergebnisse
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    for segment in segments:
        # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.words))
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
