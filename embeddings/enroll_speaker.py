from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from pathlib import Path
import os

encoder = VoiceEncoder()
embeddings_dir = "embeddings"
os.makedirs(embeddings_dir, exist_ok=True)

def enroll(speaker_name, wav_path):
    wav = preprocess_wav(Path(wav_path))
    embedding = encoder.embed_utterance(wav)
    file_path = os.path.join(embeddings_dir, f"{speaker_name}.npy")
    np.save(file_path, embedding)
    print(f"Enrollment complete for {speaker_name}.")

if __name__ == "__main__":
    speaker = input("Enter speaker name: ")
    filename = input("Enter WAV file path: ")
    enroll(speaker, filename)
