import sounddevice as sd
import numpy as np
from resemblyzer import preprocess_wav, VoiceEncoder
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

def record_audio(duration=10, fs=16000):
    print(f"\n🎤 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("✅ Recording complete.")
    return audio.flatten(), fs

def identify_live(threshold=0.75):
    embeddings_db = load_embeddings()
    
    audio, sr = record_audio(duration=5)
    wav = preprocess_wav(audio, sr)
    emb_new = encoder.embed_utterance(wav)

    best_score = -1
    best_match = None

    print("\n🔍 Computing similarities...")
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
    identify_live()
