import json
import torch
import whisper
import numpy as np
from speechbrain.inference.interfaces import foreign_class
from dotenv import load_dotenv

load_dotenv()

AUDIO_PATH = "audio/audio.wav"
VOICE_RESULTS_PATH = "audio/voice_results.json"

def transcribe_audio():
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Transcribing audio...")
    result = model.transcribe(AUDIO_PATH, word_timestamps=True)
    transcript = result["text"].strip()
    segments = result["segments"]
    print(f"Transcript: {transcript[:100]}...")
    return transcript, segments

def get_vocal_emotion():
    print("Loading SpeechBrain emotion model...")
    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )
    print("Analyzing vocal emotion...")
    out_prob, score, index, text_lab = classifier.classify_file(AUDIO_PATH)
    emotions = ["neutral", "happy", "sad", "angry"]
    probs = out_prob[0].detach().numpy().tolist()
    voice_vector = dict(zip(emotions, probs))
    dominant = text_lab[0]
    print(f"Dominant vocal emotion: {dominant}")
    return voice_vector, dominant

def run_voice_agent():
    transcript, segments = transcribe_audio()
    voice_vector, dominant_emotion = get_vocal_emotion()
    results = {
        "transcript": transcript,
        "segments": segments,
        "voice_emotion_vector": voice_vector,
        "dominant_vocal_emotion": dominant_emotion
    }
    with open(VOICE_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Voice results saved to {VOICE_RESULTS_PATH}")
    return results

if __name__ == "__main__":
    run_voice_agent()
