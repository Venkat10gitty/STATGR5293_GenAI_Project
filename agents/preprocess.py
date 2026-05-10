import os
import json
import subprocess
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

VIDEO_PATH = "interview.mp4"
AUDIO_PATH = "audio/audio.wav"
FRAMES_DIR = "frames/"
PAUSES_PATH = "audio/pauses.json"

def extract_audio():
    print("Extracting audio from video...")
    command = [
        "ffmpeg", "-y",
        "-i", VIDEO_PATH,
        "-ac", "1",
        "-ar", "16000",
        AUDIO_PATH
    ]
    subprocess.run(command, check=True)
    print("Audio extracted successfully.")

def extract_frames():
    print("Extracting frames from video...")
    command = [
        "ffmpeg", "-y",
        "-i", VIDEO_PATH,
        "-vf", "fps=1",
        os.path.join(FRAMES_DIR, "frame_%04d.jpg")
    ]
    subprocess.run(command, check=True)
    print("Frames extracted successfully.")

def detect_pauses():
    print("Detecting pauses in audio...")
    audio, sr = sf.read(AUDIO_PATH)
    frame_length = int(sr * 0.5)
    energies = []
    for i in range(0, len(audio) - frame_length, frame_length):
        frame = audio[i:i + frame_length]
        energy = float(np.sum(frame ** 2))
        energies.append(energy)
    threshold = np.mean(energies) * 0.1
    pauses = []
    for i, energy in enumerate(energies):
        if energy < threshold:
            timestamp = round(i * 0.5, 2)
            pauses.append(timestamp)
    with open(PAUSES_PATH, "w") as f:
        json.dump(pauses, f)
    print(f"Found {len(pauses)} pauses. Saved to {PAUSES_PATH}")
    return pauses

if __name__ == "__main__":
    extract_audio()
    extract_frames()
    detect_pauses()
    print("Preprocessing complete.")
