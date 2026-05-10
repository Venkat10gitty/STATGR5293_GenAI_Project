import json
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()

VOICE_RESULTS_PATH = "audio/voice_results.json"
TEXT_RESULTS_PATH = "audio/text_results.json"

EMOTION_KEYS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]

def get_roberta_emotions(transcript):
    print("Loading RoBERTa emotion model...")
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )
    print("Classifying text emotions...")
    results = classifier(transcript[:512])
    emotion_vector = {item["label"].lower(): item["score"] for item in results[0]}
    for key in EMOTION_KEYS:
        if key not in emotion_vector:
            emotion_vector[key] = 0.0
    dominant = max(emotion_vector, key=emotion_vector.get)
    print(f"Dominant text emotion: {dominant}")
    return emotion_vector, dominant

def get_vader_sentiment(transcript):
    print("Running VADER sentiment analysis...")
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(transcript)
    print(f"VADER scores: {scores}")
    return scores

def run_text_agent():
    with open(VOICE_RESULTS_PATH, "r") as f:
        voice_data = json.load(f)
    transcript = voice_data["transcript"]
    print(f"Transcript loaded: {transcript[:100]}...")
    emotion_vector, dominant_emotion = get_roberta_emotions(transcript)
    vader_scores = get_vader_sentiment(transcript)
    results = {
        "transcript": transcript,
        "text_emotion_vector": emotion_vector,
        "dominant_text_emotion": dominant_emotion,
        "vader_sentiment": vader_scores
    }
    with open(TEXT_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Text results saved to {TEXT_RESULTS_PATH}")
    return results

if __name__ == "__main__":
    run_text_agent()
