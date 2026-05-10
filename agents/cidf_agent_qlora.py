import json
import numpy as np
import torch
import re
from scipy.spatial.distance import cosine

# File paths for agent results
FACE_RESULTS_PATH = "audio/face_results.json"
VOICE_RESULTS_PATH = "audio/voice_results.json"
TEXT_RESULTS_PATH = "audio/text_results.json"
CIDF_RESULTS_PATH = "audio/cidf_results.json"

# Standard MELD emotion categories
MELD_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# DeepFace label to MELD mapping
DF_TO_MELD = {
    "angry": "anger", "disgust": "disgust", "fear": "fear",
    "happy": "joy", "sad": "sadness", "surprise": "surprise", "neutral": "neutral"
}

# SpeechBrain wav2vec2 IEMOCAP label to MELD mapping
SUPERB_TO_MELD = {
    "neu": "neutral", "hap": "joy", "ang": "anger", "sad": "sadness",
    "neutral": "neutral", "happy": "joy", "angry": "anger"
}

# RoBERTa label to MELD mapping
ROBERTA_TO_MELD = {
    "anger": "anger", "disgust": "disgust", "fear": "fear",
    "joy": "joy", "neutral": "neutral", "sadness": "sadness", "surprise": "surprise",
    "happy": "joy", "sad": "sadness", "angry": "anger"
}

def build_emotion_vector(emotion_probs, label_map):
    """Convert raw model probabilities to normalized 7d MELD emotion vector."""
    vec = np.zeros(7, dtype=np.float32)
    for raw_label, prob in emotion_probs.items():
        meld = label_map.get(raw_label.lower())
        if meld and meld in MELD_EMOTIONS:
            vec[MELD_EMOTIONS.index(meld)] += float(prob)
    if vec.sum() > 0:
        vec = vec / vec.sum()
    return vec

def compute_mas(vec_a, vec_b):
    """Compute Modal Agreement Score as cosine similarity between two emotion vectors."""
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return float(1.0 - cosine(vec_a, vec_b))

def get_dominant_conflict(mas_fv, mas_ft, mas_vt):
    """Identify the modality pair with lowest agreement — the dominant conflict."""
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

def run_qlora_inference(face_vec, voice_vec, text_vec, mas_fv, mas_ft, mas_vt, model, tokenizer):
    """Run QLoRA fine-tuned Qwen2-VL-7B to detect emotional incongruence.
    Uses detailed structured prompt for consistent results.
    """
    prompt = f"""You are an emotional incongruence detection model.
Given the following multimodal emotion analysis:
- Face emotion vector: {dict(zip(MELD_EMOTIONS, face_vec.tolist()))}
- Voice emotion vector: {dict(zip(MELD_EMOTIONS, voice_vec.tolist()))}
- Text emotion vector: {dict(zip(MELD_EMOTIONS, text_vec.tolist()))}
- Modal Agreement Score (Face-Voice): {mas_fv:.4f}
- Modal Agreement Score (Face-Text): {mas_ft:.4f}
- Modal Agreement Score (Voice-Text): {mas_vt:.4f}

Is this utterance emotionally incongruent?
Answer with a JSON object containing:
- incongruent: true or false
- incongruence_score: float between 0 and 1
- dominant_conflict: the most conflicting modality pair
- confidence: float between 0 and 1
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"QLoRA response: {response}")

    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            score = float(result.get("incongruence_score", 0.5))
        else:
            score = 1.0 - float(np.mean([mas_fv, mas_ft, mas_vt]))
    except:
        score = 1.0 - float(np.mean([mas_fv, mas_ft, mas_vt]))

    return round(score, 4)

def run_cidf_agent(model=None, tokenizer=None):
    """Main CIDF agent — loads emotion results, computes MAS, runs QLoRA inference."""

    # Load results from all three modality agents
    with open(FACE_RESULTS_PATH, "r") as f:
        face_data = json.load(f)
    with open(VOICE_RESULTS_PATH, "r") as f:
        voice_data = json.load(f)
    with open(TEXT_RESULTS_PATH, "r") as f:
        text_data = json.load(f)

    # Build normalized 7d emotion vectors for each modality
    face_vec = build_emotion_vector(face_data["face_emotion_vector"], DF_TO_MELD)
    voice_vec = build_emotion_vector(voice_data["voice_emotion_vector"], SUPERB_TO_MELD)
    text_vec = build_emotion_vector(text_data["text_emotion_vector"], ROBERTA_TO_MELD)

    # Compute pairwise Modal Agreement Scores
    mas_fv = compute_mas(face_vec, voice_vec)
    mas_ft = compute_mas(face_vec, text_vec)
    mas_vt = compute_mas(voice_vec, text_vec)

    mas_scores, dominant_conflict_pair, conflict_description = get_dominant_conflict(
        mas_fv, mas_ft, mas_vt
    )
    print(f"MAS Scores: {mas_scores}")

    # Run QLoRA inference if model is available else use MAS fallback
    if model is not None and tokenizer is not None:
        incongruence_score = run_qlora_inference(
            face_vec, voice_vec, text_vec,
            mas_fv, mas_ft, mas_vt,
            model, tokenizer
        )
        model_type = "qlora_cidf"
        print(f"QLoRA Incongruence Score: {incongruence_score}")
    else:
        incongruence_score = round(1.0 - float(np.mean([mas_fv, mas_ft, mas_vt])), 4)
        model_type = "mas_fallback"
        print(f"Fallback Incongruence Score: {incongruence_score}")

    results = {
        "mas_scores": mas_scores,
        "incongruence_score": incongruence_score,
        "is_incongruent": incongruence_score > 0.5,
        "dominant_conflict_pair": dominant_conflict_pair,
        "conflict_description": conflict_description,
        "dominant_facial_emotion": face_data["dominant_facial_emotion"],
        "dominant_vocal_emotion": voice_data["dominant_vocal_emotion"],
        "dominant_text_emotion": text_data["dominant_text_emotion"],
        "model_type": model_type
    }

    with open(CIDF_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"CIDF results saved to {CIDF_RESULTS_PATH}")
    return results
