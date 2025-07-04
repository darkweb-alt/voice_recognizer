import streamlit as st
import numpy as np
import sounddevice as sd
from resemblyzer import preprocess_wav, VoiceEncoder
import os

# Initialize encoder
encoder = VoiceEncoder()
embeddings_dir = "embeddings"

# Function to load stored embeddings
def load_embeddings():
    embeddings = {}
    for file in os.listdir(embeddings_dir):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            embedding = np.load(os.path.join(embeddings_dir, file))
            embeddings[name] = embedding
    return embeddings

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Record audio
def record_audio(duration=5, fs=16000):
    st.info(f"Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    st.success("Recording complete.")
    return audio.flatten(), fs

# Streamlit UI
st.title("🎙️ Voice Identification System")

if st.button("Record and Identify"):
    embeddings_db = load_embeddings()

    # Record 5 seconds audio
    audio, sr = record_audio(duration=5)

    # Preprocess and embed
    wav = preprocess_wav(audio, sr)
    emb_new = encoder.embed_utterance(wav)

    # Compute similarities
    best_score = -1
    best_match = None

    st.write("🔍 Comparing with enrolled speakers...")
    for name, emb_ref in embeddings_db.items():
        score = cosine_similarity(emb_ref, emb_new)
        st.write(f"Similarity with {name}: {score:.3f}")
        if score > best_score:
            best_score = score
            best_match = name

    # Threshold
    threshold = 0.75

    # Show result
    if best_score >= threshold:
        
        st.success(f"✅ Hello, you are **{best_match}** (Score: {best_score:.3f})")
    else:
        st.error(f"❌ Unknown Speaker (Best Score: {best_score:.3f})")
if(best_match=="Pitambar Yadav"):
    print("boss came")
    