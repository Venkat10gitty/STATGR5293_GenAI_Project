import json
import numpy as np
import torch
import pickle
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

load_dotenv()

FACE_RESULTS_PATH = "audio/face_results.json"
VOICE_RESULTS_PATH = "audio/voice_results.json"
TEXT_RESULTS_PATH = "audio/text_results.json"
CIDF_RESULTS_PATH = "audio/cidf_results.json"
MLP_MODEL_PATH = "models/mlp_fusion.pt"
SCALER_PATH = "models/scaler.pkl"

MELD_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

DF_TO_MELD = {
    "angry": "anger", "disgust": "disgust", "fear": "fear",
    "happy": "joy", "sad": "sadness", "surprise": "surprise", "neutral": "neutral"
}

SUPERB_TO_MELD = {
    "neu": "neutral", "hap": "joy", "ang": "anger", "sad": "sadness",
    "neutral": "neutral", "happy": "joy", "angry": "anger", "sad": "sadness"
}

ROBERTA_TO_MELD = {
    "anger": "anger", "disgust": "disgust", "fear": "fear",
    "joy": "joy", "neutral": "neutral", "sadness": "sadness", "surprise": "surprise",
    "happy": "joy", "sad": "sadness", "angry": "anger"
}

class MLPFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(24, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def build_emotion_vector(emotion_probs, label_map):
    vec = np.zeros(7, dtype=np.float32)
    for raw_label, prob in emotion_probs.items():
        meld = label_map.get(raw_label.lower())
        if meld and meld in MELD_EMOTIONS:
            vec[MELD_EMOTIONS.index(meld)] += float(prob)
    if vec.sum() > 0:
        vec = vec / vec.sum()
    return vec

def compute_mas(vec_a, vec_b):
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return float(1.0 - cosine(vec_a, vec_b))

def build_feature_vector(face_probs, voice_probs, text_probs):
    face_vec = build_emotion_vector(face_probs, DF_TO_MELD)
    voice_vec = build_emotion_vector(voice_probs, SUPERB_TO_MELD)
    text_vec = build_emotion_vector(text_probs, ROBERTA_TO_MELD)

    mas_fv = compute_mas(face_vec, voice_vec)
    mas_ft = compute_mas(face_vec, text_vec)
    mas_vt = compute_mas(voice_vec, text_vec)

    return np.concatenate([
        face_vec,
        voice_vec,
        text_vec,
        [mas_fv, mas_ft, mas_vt]
    ]).astype(np.float32), mas_fv, mas_ft, mas_vt

def get_dominant_conflict(mas_fv, mas_ft, mas_vt):
    mas_scores = {
        "face_voice": round(float(mas_fv), 4),
        "face_text": round(float(mas_ft), 4),
        "voice_text": round(float(mas_vt), 4)
    }
    min_pair = min(mas_scores, key=mas_scores.get)
    conflict_map = {
        "face_voice": "Face and Voice are conflicting",
        "face_text": "Face and Text are conflicting",
        "voice_text": "Voice and Text are conflicting"
    }
    return mas_scores, min_pair, conflict_map[min_pair]

def load_mlp_model():
    try:
        model = MLPFusion()
        state_dict = torch.load(MLP_MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("MLP model loaded successfully.")
        return model, scaler
    except Exception as e:
        print(f"Could not load MLP model: {e}. Using dummy fallback.")
        return None, None

def predict_incongruence(model, scaler, features):
    if model is None or scaler is None:
        avg_mas = float(np.mean(features[21:24]))
        return round(1.0 - avg_mas, 4), False
    try:
        x = scaler.transform([features])
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        is_incongruent = prob > 0.5
        return round(prob, 4), is_incongruent
    except Exception as e:
        print(f"MLP inference error: {e}. Using fallback.")
        avg_mas = float(np.mean(features[21:24]))
        return round(1.0 - avg_mas, 4), False

def run_cidf_agent():
    with open(FACE_RESULTS_PATH, "r") as f:
        face_data = json.load(f)
    with open(VOICE_RESULTS_PATH, "r") as f:
        voice_data = json.load(f)
    with open(TEXT_RESULTS_PATH, "r") as f:
        text_data = json.load(f)

    face_probs = face_data["face_emotion_vector"]
    voice_probs = voice_data["voice_emotion_vector"]
    text_probs = text_data["text_emotion_vector"]

    features, mas_fv, mas_ft, mas_vt = build_feature_vector(
        face_probs, voice_probs, text_probs
    )

    mas_scores, dominant_conflict_pair, conflict_description = get_dominant_conflict(
        mas_fv, mas_ft, mas_vt
    )

    print(f"MAS Scores: {mas_scores}")

    model, scaler = load_mlp_model()
    incongruence_score, is_incongruent = predict_incongruence(model, scaler, features)

    print(f"Incongruence Score: {incongruence_score}")
    print(f"Is Incongruent: {is_incongruent}")
    print(f"Dominant Conflict: {conflict_description}")

    results = {
        "mas_scores": mas_scores,
        "incongruence_score": incongruence_score,
        "is_incongruent": is_incongruent,
        "dominant_conflict_pair": dominant_conflict_pair,
        "conflict_description": conflict_description,
        "dominant_facial_emotion": face_data["dominant_facial_emotion"],
        "dominant_vocal_emotion": voice_data["dominant_vocal_emotion"],
        "dominant_text_emotion": text_data["dominant_text_emotion"],
        "model_type": "mlp_fusion"
    }

    with open(CIDF_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"CIDF results saved to {CIDF_RESULTS_PATH}")
    return results

if __name__ == "__main__":
    run_cidf_agent()
