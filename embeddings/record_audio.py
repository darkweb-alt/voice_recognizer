import sounddevice as sd
from scipy.io.wavfile import write

def record(filename, duration=5, fs=16000):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    print(f"Saved recording as {filename}")

if __name__ == "__main__":
    record("test.wav", duration=10)
