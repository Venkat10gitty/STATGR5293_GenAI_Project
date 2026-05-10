import json
import os
from openai import OpenAI
from elevenlabs import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

CIDF_RESULTS_PATH = "audio/cidf_results.json"
COACHING_TEXT_PATH = "outputs/coaching_message.txt"
COACHING_AUDIO_PATH = "outputs/coaching_audio.mp3"

openai_client = OpenAI()
eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def generate_coaching_message(cidf_results):
    print("Generating coaching message with GPT-4o...")
    prompt = f"""
You are a professional communication coach. A person is being analyzed for emotional incongruence.

Analysis results:
- Face emotion: {cidf_results['dominant_facial_emotion']}
- Voice emotion: {cidf_results['dominant_vocal_emotion']}
- Text emotion: {cidf_results['dominant_text_emotion']}
- Incongruence score: {cidf_results['incongruence_score']} (0=congruent, 1=very incongruent)
- Main conflict: {cidf_results['conflict_description']}
- MAS scores: {cidf_results['mas_scores']}

Write a short, specific, empathetic coaching suggestion (2-3 sentences maximum) that:
1. Identifies the specific mismatch detected
2. Gives one concrete actionable tip to align the conflicting channels
3. Is encouraging and professional in tone

Do not use technical terms like MAS or incongruence score. Speak directly to the person.
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    message = response.choices[0].message.content.strip()
    print(f"Coaching message: {message}")
    return message

def text_to_speech(coaching_message):
    print("Converting coaching message to speech with ElevenLabs...")
    audio = eleven_client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text=coaching_message,
        model_id="eleven_turbo_v2_5"
    )
    with open(COACHING_AUDIO_PATH, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    print(f"Coaching audio saved to {COACHING_AUDIO_PATH}")

def run_coach_agent():
    with open(CIDF_RESULTS_PATH, "r") as f:
        cidf_results = json.load(f)
    coaching_message = generate_coaching_message(cidf_results)
    with open(COACHING_TEXT_PATH, "w") as f:
        f.write(coaching_message)
    text_to_speech(coaching_message)
    print("Coach agent complete.")
    return coaching_message

if __name__ == "__main__":
    run_coach_agent()
