import gradio as gr
import os
import shutil
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from preprocess import extract_audio, extract_frames, detect_pauses
from voice_agent import run_voice_agent
from face_agent import run_face_agent
from text_agent import run_text_agent
from cidf_agent import run_cidf_agent
from coach_agent import run_coach_agent

openai_client = OpenAI()

EMOTION_EMOJI = {
    "happy": "😊", "sad": "😢", "angry": "😡",
    "neutral": "😐", "fear": "😨", "disgust": "🤢",
    "surprise": "😲", "joy": "😊", "sadness": "😢",
    "anger": "😡"
}

def get_emoji(emotion):
    return EMOTION_EMOJI.get(emotion.lower(), "😐")

def generate_general_suggestion(cidf_results):
    prompt = f"""
You are a communication coach. Give ONE short punchy tip (maximum 2 sentences) for someone whose:
- Face shows: {cidf_results['dominant_facial_emotion']}
- Voice shows: {cidf_results['dominant_vocal_emotion']}
- Words show: {cidf_results['dominant_text_emotion']}
- Main conflict: {cidf_results['conflict_description']}
Be direct, practical, energetic. No fluff. Start with a strong action verb.
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60
    )
    return response.choices[0].message.content.strip()

def run_truesignal(video_path):
    if video_path is None:
        yield ("Please upload a video first.", "", "", "", "", "", "", "", "", None)
        return
    try:
        shutil.copy(video_path, "interview.mp4")
        yield ("⚙️ Step 1/5: Preprocessing video...", "—", "—", "—", "—", "—", "—", "—", "—", None)
        extract_audio()
        extract_frames()
        detect_pauses()

        yield ("🎙️ Step 2/5: Analyzing voice...", "Processing...", "—", "—", "—", "—", "—", "—", "—", None)
        voice_results = run_voice_agent()
        v_emotion = voice_results['dominant_vocal_emotion']
        v_emoji = get_emoji(v_emotion)

        yield (f"👁️ Step 3/5: Analyzing face...", f"{v_emoji} {v_emotion.upper()}", "Processing...", "—", "—", "—", "—", "—", "—", None)
        face_results = run_face_agent()
        f_emotion = face_results['dominant_facial_emotion']
        f_emoji = get_emoji(f_emotion)

        yield (f"📋 Step 4/5: Analyzing text...", f"{v_emoji} {v_emotion.upper()}", f"{f_emoji} {f_emotion.upper()}", "Processing...", "—", "—", "—", "—", "—", None)
        text_results = run_text_agent()
        t_emotion = text_results['dominant_text_emotion']
        t_emoji = get_emoji(t_emotion)

        yield ("🧠 Step 5/5: Computing incongruence...", f"{v_emoji} {v_emotion.upper()}", f"{f_emoji} {f_emotion.upper()}", f"{t_emoji} {t_emotion.upper()}", "Computing...", "—", "—", "—", "—", None)
        cidf_results = run_cidf_agent()

        mas = cidf_results["mas_scores"]
        score = cidf_results["incongruence_score"]
        conflict = cidf_results["conflict_description"]

        mas_display = f"🎙️ Face ↔ Voice:   {mas['face_voice']:.4f}\n👁️ Face ↔ Text:    {mas['face_text']:.4f}\n📋 Voice ↔ Text:  {mas['voice_text']:.4f}"

        if score >= 0.7:
            score_label = f"🔴 {score:.4f} — HIGH INCONGRUENCE"
        elif score >= 0.5:
            score_label = f"🟡 {score:.4f} — MODERATE INCONGRUENCE"
        else:
            score_label = f"🟢 {score:.4f} — LOW INCONGRUENCE"

        yield ("🎤 Generating coaching...", f"{v_emoji} {v_emotion.upper()}", f"{f_emoji} {f_emotion.upper()}", f"{t_emoji} {t_emotion.upper()}", mas_display, score_label, conflict, "Generating...", "Generating...", None)

        general_suggestion = generate_general_suggestion(cidf_results)
        coaching_message = run_coach_agent()
        audio_path = "outputs/coaching_audio.mp3"

        yield ("✅ Analysis Complete!", f"{v_emoji} {v_emotion.upper()}", f"{f_emoji} {f_emotion.upper()}", f"{t_emoji} {t_emotion.upper()}", mas_display, score_label, conflict, general_suggestion, coaching_message, audio_path)

    except Exception as e:
        yield (f"❌ Error: {str(e)}", "—", "—", "—", "—", "—", "—", "—", "—", None)

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

body, .gradio-container {
    background: #eef2f7 !important;
    font-family: 'Inter', sans-serif !important;
    color: #1a1a2e !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
}

#truesignal-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f3460 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

#truesignal-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #64b5f6, #ce93d8, #80cbc4, transparent);
}

#ts-badge {
    background: rgba(100,181,246,0.12);
    border: 1px solid rgba(100,181,246,0.3);
    border-radius: 20px;
    padding: 5px 16px;
    font-size: 10px;
    color: #90caf9;
    letter-spacing: 2px;
    text-transform: uppercase;
    display: inline-block;
    margin-bottom: 16px;
}

#ts-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2em;
    font-weight: 700;
    letter-spacing: -2px;
    line-height: 1;
    margin: 0;
}

#ts-title .t1 { color: #ffffff; }
#ts-title .t2 {
    background: linear-gradient(135deg, #64b5f6, #ce93d8, #80cbc4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

#ts-subtitle {
    font-size: 0.72em;
    color: #78909c;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 12px;
    display: block;
}

#ts-divider {
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, #64b5f6, #ce93d8);
    margin: 14px auto 0;
    border-radius: 2px;
}

label span {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72em !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #78909c !important;
}

textarea {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #1a1a2e !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9em !important;
    padding: 12px 14px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}

.upload-zone {
    border: 2px dashed #90caf9 !important;
    border-radius: 14px !important;
    background: #ffffff !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    min-height: 180px !important;
}

.analyze-btn {
    background: linear-gradient(135deg, #0f172a, #1e3a5f, #2d5986) !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1em !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(30,58,95,0.3) !important;
    transition: all 0.3s ease !important;
    margin-top: 8px !important;
}

.analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(30,58,95,0.4) !important;
}

.status-box textarea {
    color: #1e88e5 !important;
    text-align: center !important;
    font-weight: 600 !important;
}

.tip-box textarea {
    color: #4e342e !important;
    font-size: 0.95em !important;
    line-height: 1.7 !important;
    border-left: 4px solid #f9a825 !important;
    background: #fffde7 !important;
    min-height: 100px !important;
}

.voice-box textarea {
    color: #1e88e5 !important;
    font-weight: 700 !important;
    font-size: 1.1em !important;
    text-align: center !important;
    border-top: 3px solid #1e88e5 !important;
}

.face-box textarea {
    color: #7b1fa2 !important;
    font-weight: 700 !important;
    font-size: 1.1em !important;
    text-align: center !important;
    border-top: 3px solid #7b1fa2 !important;
}

.text-box textarea {
    color: #00897b !important;
    font-weight: 700 !important;
    font-size: 1.1em !important;
    text-align: center !important;
    border-top: 3px solid #00897b !important;
}

.mas-box textarea {
    color: #1e88e5 !important;
    line-height: 2.2 !important;
    font-size: 0.88em !important;
}

.score-box textarea {
    font-size: 1.2em !important;
    font-weight: 800 !important;
    text-align: center !important;
    color: #e53935 !important;
}

.conflict-box textarea {
    color: #bf360c !important;
    font-weight: 600 !important;
    border-left: 4px solid #fb8c00 !important;
}

.coaching-box textarea {
    color: #1a1a2e !important;
    font-size: 0.95em !important;
    line-height: 1.8 !important;
    border-left: 4px solid #00897b !important;
    background: #f0faf8 !important;
}

.footer-area {
    text-align: center;
    padding: 20px 0 8px;
    color: #90a4ae;
    font-size: 0.75em;
    letter-spacing: 1px;
    border-top: 1px solid #e2e8f0;
    margin-top: 24px;
    font-family: 'Inter', sans-serif;
}
"""

with gr.Blocks(css=css, title="TrueSignal") as app:

    gr.HTML("""
    <div id="truesignal-header">
        <div id="ts-badge">⚡ Live Analysis System</div>
        <h1 id="ts-title"><span class="t1">True</span><span class="t2">Signal</span></h1>
        <span id="ts-subtitle">Real-Time Multimodal Emotional Incongruence Detection</span>
        <div id="ts-divider"></div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="📹 Upload Video", elem_classes=["upload-zone"])
            analyze_btn = gr.Button("⚡ Analyze", elem_classes=["analyze-btn"])
            status_output = gr.Textbox(
                label="Pipeline Status",
                interactive=False,
                elem_classes=["status-box"]
            )
            suggestion_output = gr.Textbox(
                label="💡 Quick Tip",
                interactive=False,
                lines=4,
                elem_classes=["tip-box"]
            )

        with gr.Column(scale=1):
            with gr.Row():
                voice_output = gr.Textbox(label="🎙️ Voice", interactive=False, elem_classes=["voice-box"])
                face_output = gr.Textbox(label="👁️ Face", interactive=False, elem_classes=["face-box"])
                text_output = gr.Textbox(label="📋 Text", interactive=False, elem_classes=["text-box"])

            mas_output = gr.Textbox(
                label="📊 Modal Agreement Scores",
                interactive=False,
                lines=3,
                max_lines=3,
                elem_classes=["mas-box"]
            )

            with gr.Row():
                score_output = gr.Textbox(
                    label="🔴 Incongruence Score",
                    interactive=False,
                    elem_classes=["score-box"]
                )
                conflict_output = gr.Textbox(
                    label="⚡ Dominant Conflict",
                    interactive=False,
                    elem_classes=["conflict-box"]
                )

    with gr.Row():
        with gr.Column(scale=2):
            coaching_output = gr.Textbox(
                label="💬 Coaching Message",
                interactive=False,
                lines=4,
                elem_classes=["coaching-box"]
            )
        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="🔊 Spoken Coaching",
                interactive=False
            )

    gr.HTML('<div class="footer-area">TrueSignal &nbsp;·&nbsp; STAT GR5293 &nbsp;·&nbsp; Columbia University &nbsp;·&nbsp; Spring 2026</div>')

    analyze_btn.click(
        fn=run_truesignal,
        inputs=[video_input],
        outputs=[
            status_output,
            voice_output,
            face_output,
            text_output,
            mas_output,
            score_output,
            conflict_output,
            suggestion_output,
            coaching_output,
            audio_output
        ]
    )

if __name__ == "__main__":
    app.launch(share=False)
