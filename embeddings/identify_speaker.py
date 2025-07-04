from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from pathlib import Path
import os

encoder = VoiceEncoder()
embeddings_dir = "embeddings"

def load_embeddings():
    embeddings = {}
    for file in os.listdir(embeddings_dir):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            embedding = np.load(os.path.join(embeddings_dir, file))
            embeddings[name] = embedding
    return embeddings

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def identify(wav_path, threshold=0.75):
    embeddings_db = load_embeddings()
    wav = preprocess_wav(Path(wav_path))
    emb_new = encoder.embed_utterance(wav)

    best_score = -1
    best_match = None
    for name, emb_ref in embeddings_db.items():
        score = cosine_similarity(emb_ref, emb_new)
        print(f"Similarity with {name}: {score:.3f}")
        if score > best_score:
            best_score = score
            best_match = name

    if best_score >= threshold:
        print(f"\n✅ Hello 👋 You are {best_match}. (Score: {best_score:.3f})")
    else:
        print(f"\n❌ Unknown Speaker. Best score: {best_score:.3f}")

if __name__ == "__main__":
    file = input("Enter WAV file path for identification: ")
    identify(file)
