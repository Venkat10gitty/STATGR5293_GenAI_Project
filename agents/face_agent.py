import os
import json
import base64
import numpy as np
from deepface import DeepFace
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

FRAMES_DIR = "frames/"
PAUSES_PATH = "audio/pauses.json"
FACE_RESULTS_PATH = "audio/face_results.json"

client = OpenAI()

EMOTION_KEYS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def convert_floats(obj):
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj

def analyze_frames_deepface():
    print("Analyzing frames with DeepFace...")
    frame_files = sorted([
        f for f in os.listdir(FRAMES_DIR)
        if f.endswith(".jpg")
    ])
    emotion_accumulator = {k: 0.0 for k in EMOTION_KEYS}
    valid_frames = 0
    frame_results = []
    for frame_file in frame_files:
        frame_path = os.path.join(FRAMES_DIR, frame_file)
        try:
            result = DeepFace.analyze(
                img_path=frame_path,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )
            emotions = result[0]["emotion"]
            for k in EMOTION_KEYS:
                emotion_accumulator[k] += float(emotions.get(k, 0.0))
            valid_frames += 1
            frame_results.append({
                "frame": frame_file,
                "emotions": convert_floats(emotions)
            })
        except Exception as e:
            continue
    if valid_frames > 0:
        for k in EMOTION_KEYS:
            emotion_accumulator[k] /= valid_frames
    dominant = max(emotion_accumulator, key=emotion_accumulator.get)
    print(f"Dominant facial emotion: {dominant}")
    return emotion_accumulator, dominant, frame_results

def analyze_pause_frame_gpt4v(frame_path):
    print(f"Analyzing pause frame with GPT-4V: {frame_path}")
    with open(frame_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Analyze the facial emotion in this image. Return a JSON object with keys: dominant_emotion, confidence, and description. dominant_emotion must be one of: angry, disgust, fear, happy, sad, surprise, neutral."
                    }
                ]
            }
        ],
        max_tokens=200
    )
    content = response.choices[0].message.content
    try:
        content_clean = content.strip().replace("```json", "").replace("```", "")
        gpt4v_result = json.loads(content_clean)
    except:
        gpt4v_result = {"dominant_emotion": "neutral", "confidence": 0.5, "description": content}
    return gpt4v_result

def get_pause_frame_paths():
    with open(PAUSES_PATH, "r") as f:
        pauses = json.load(f)
    frame_files = sorted([
        f for f in os.listdir(FRAMES_DIR)
        if f.endswith(".jpg")
    ])
    pause_frames = []
    for pause_time in pauses[:3]:
        frame_index = min(int(pause_time), len(frame_files) - 1)
        if frame_index < len(frame_files):
            pause_frames.append(os.path.join(FRAMES_DIR, frame_files[frame_index]))
    return pause_frames

def run_face_agent():
    face_vector, dominant_emotion, frame_results = analyze_frames_deepface()
    pause_frames = get_pause_frame_paths()
    gpt4v_results = []
    for frame_path in pause_frames:
        gpt4v_result = analyze_pause_frame_gpt4v(frame_path)
        gpt4v_results.append(gpt4v_result)
    results = {
        "face_emotion_vector": convert_floats(face_vector),
        "dominant_facial_emotion": dominant_emotion,
        "gpt4v_pause_analysis": gpt4v_results
    }
    with open(FACE_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Face results saved to {FACE_RESULTS_PATH}")
    return results

if __name__ == "__main__":
    run_face_agent()
