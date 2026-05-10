# TrueSignal
## Real-Time Multimodal Emotional Incongruence Detection

**STAT GR5293 — Generative AI with LLMs**
Columbia University, Spring 2026

**Team:** Sai Venkata Teja Bacham (sb5206), Sowmya Yerraguntla (sy3330), Sai Teja Dusari (sd3990)

---

## What This Project Does

People communicate through three channels at once: their face, their voice, and their words. In high-stakes conversations, these channels frequently conflict. A candidate says they are confident while their voice is tight and their face is composed. A speaker delivers enthusiastic words in a flat monotone. Listeners pick up on this disconnect and walk away thinking something felt off, even when they cannot articulate what.

This phenomenon is called emotional incongruence, and no existing tool detects it across all three channels simultaneously. Speech coaches analyze voice. Presentation tools analyze words. Facial analysis software reads expressions. Each works in isolation.

TrueSignal processes face, voice, and text in parallel through five specialized LangChain agents, computes a pairwise agreement score across all three modality pairs before any fusion takes place, and delivers spoken coaching during natural speech pauses when a conflict is detected. The full pipeline runs for under seven dollars in total API costs.

---

## Results

Evaluated on the held-out MELD test set of 2,615 clips:

| Model | F1 | AUROC | Type |
|---|---|---|---|
| B1 VADER text-only | 0.3038 | — | Baseline |
| B2 Voice-only wav2vec2 | 0.8055 | — | Baseline |
| B4 Majority class | 0.8070 | — | Baseline |
| MLP Fusion 24d | 0.8997 | 0.9291 | Our model |
| CIDF QLoRA Qwen2-VL-7B | 0.9935 | 0.9895 | Our model |

Fine-tuning gain over MLP: +9.38 F1 points
MAS algorithm contribution: +0.0055 F1 points (consistent across all 2,615 test clips)
Cohen's Kappa for label validation: 0.82 (almost perfect agreement, Landis and Koch scale)

Out-of-distribution test: Both models were run on a real interview video with no connection to the MELD training data. The MLP scored 0.9981 (HIGH INCONGRUENCE). The QLoRA scored 0.5907 (MODERATE INCONGRUENCE). Both independently identified face-voice as the dominant conflict pair.

---

## Novel Contribution: Modal Agreement Score

Before any feature fusion takes place, TrueSignal computes three pairwise cosine similarities between the emotion vectors from each modality pair:

```
MAS(A, B) = 1 - cosine_distance(vector_A, vector_B)
```

A score near 0.0 means the two modalities are in maximum disagreement. A score near 1.0 means they are perfectly aligned. Three scores are computed per clip: face-voice, face-text, and voice-text. These form the final three dimensions of the 24-dimensional feature vector fed to both the MLP and the QLoRA model.

No prior system computes pairwise cross-modal agreement explicitly before fusion. This is TrueSignal's core algorithmic contribution.

---

## Architecture

The system consists of five LangChain agents connected through an LCEL pipeline with RunnableBranch conditional routing.

**Face Agent** reads frames from the uploaded video. DeepFace runs on every frame at one frame per second. On detected speech pauses, GPT-4V performs a deeper facial analysis. Output is a 7-class emotion vector over the MELD emotion labels.

**Voice Agent** transcribes the audio using OpenAI Whisper. Energy-based silence detection identifies speech pauses. SpeechBrain wav2vec2 (trained on IEMOCAP) classifies vocal emotion into four classes: neutral, happy, angry, sad. Output is a 7-dimensional vector, zero-padded for the three classes not covered by the IEMOCAP model.

**Text Agent** runs the transcript through j-hartmann/emotion-english-distilroberta-base, which outputs all seven MELD emotion classes directly. VADER is also run for compound sentiment. Output is a 7-class emotion vector.

**CIDF Agent** (Cross-Modal Incongruence Detection Framework) takes the three emotion vectors, computes the three MAS scores, concatenates everything into a 24-dimensional feature vector, and runs either the MLP fusion model (for local CPU inference) or the QLoRA fine-tuned Qwen2-VL-7B (for GPU inference on Colab).

**Coach Agent** triggers only when the incongruence score exceeds 0.50. GPT-4o generates a personalized coaching message targeting the specific conflicting modality pair. ElevenLabs (eleven_turbo_v2_5) converts the text to speech and plays it aloud.

---

## Repository Structure

```
STATGR5293_GenAI_Project/
│
├── app.py                      Gradio UI — MLP version, runs locally on CPU
├── app_qlora.py                Gradio UI — QLoRA version, runs on Colab A100
├── pipeline.py                 Full LangChain LCEL pipeline
├── preprocess.py               Audio extraction, frame extraction, pause detection
├── voice_agent.py              Whisper transcription + SpeechBrain vocal emotion
├── face_agent.py               DeepFace frame emotion + GPT-4V pause analysis
├── text_agent.py               RoBERTa emotion + VADER sentiment
├── cidf_agent.py               MAS computation + MLP fusion inference
├── cidf_agent_qlora.py         MAS computation + QLoRA Qwen2-VL-7B inference
├── coach_agent.py              GPT-4o coaching generation + ElevenLabs TTS
├── Truesignal_QLoRA.ipynb      Colab notebook — full QLoRA inference pipeline
├── TrueSignal_Pipeline.ipynb   Colab notebook — data extraction, labeling, training
│
├── pipeline/
│   ├── extract_clips.py        MELD clip extraction (frames + audio + transcripts)
│   └── label_construction.py  Incongruence label construction + MAS computation
│
├── models/
│   ├── mlp_fusion.py           MLP model definition, training, and inference
│   ├── mlp_fusion.pt           Trained MLP weights — F1 = 0.8997
│   └── scaler.pkl              StandardScaler fitted on MELD train split
│
├── eval/
│   └── ablations.py            All four ablation axes
│
├── results/
│   ├── mlp_results.json        MLP test results
│   ├── cidf_results.json       QLoRA CIDF test results
│   ├── ablation_results.json   Full ablation study results
│   ├── baseline_results.json   All baseline results
│   └── kappa_result.json       Cohen's Kappa validation result
│
├── data/
│   └── sample_labels.csv       200-clip sample of the full 13,715-clip label set
│
├── tests/
│   └── test_truesignal.py      Unit tests — 15 tests across 4 test classes
│
└── requirements.txt
```

---

## Setup

### Running the Gradio Demo Locally

The demo was developed and tested on Mac M4 Pro running Python 3.11 via conda. Python 3.13 is not supported due to incompatibilities with the ML libraries used.

**Step 1.** Create a conda environment with Python 3.11:

```bash
conda create -n truesignal python=3.11
conda activate truesignal
```

**Step 2.** Install dependencies:

```bash
pip install -r requirements.txt
pip install tf-keras
```

**Step 3.** Set your API keys. Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
```

**Step 4.** The MLP checkpoint is already in the repository under `models/`. Verify that `models/mlp_fusion.pt` and `models/scaler.pkl` are present before running.

**Step 5.** Launch the MLP version:

```bash
python app.py
```

The Gradio interface will open in your browser. Upload any video file and click Analyze.

### Running the QLoRA Version on Google Colab

The QLoRA version requires a GPU with at least 40GB VRAM. Open `Truesignal_QLoRA.ipynb` in Google Colab, select A100 as the runtime accelerator, and run the cells in order. The QLoRA adapter must be downloaded separately — see the checkpoint section below.

### Reproducing the Full Training Pipeline

**Step 1.** Open `TrueSignal_Pipeline.ipynb` in Google Colab with an A100 runtime.

**Step 2.** Download the MELD dataset:

```bash
python data/download_meld.py --output_dir /path/to/meld_raw
```

This downloads approximately 11GB and takes 5 to 10 minutes.

**Step 3.** Extract frames, audio, and transcripts from all 13,715 clips:

```bash
python pipeline/extract_clips.py \
    --meld_raw  /path/to/meld_raw \
    --meld_proc /path/to/meld_processed \
    --split     all
```

This takes approximately 8 hours on an A100. The script is resume-safe and can be interrupted and restarted without losing progress.

**Step 4.** Build incongruence labels:

```bash
python pipeline/label_construction.py \
    --meld_proc  /path/to/meld_processed \
    --labels_dir /path/to/labels
```

Also approximately 8 hours. Outputs are written incrementally so partial progress is never lost.

**Step 5.** Train the MLP fusion model:

```bash
python models/mlp_fusion.py --train \
    --labels_csv /path/to/labels/all_labels.csv \
    --output_dir models/
```

Completes in under 5 minutes. Saves `mlp_fusion.pt` and `scaler.pkl`.

**Step 6.** Run the ablation study:

```bash
python eval/ablations.py \
    --labels_csv  /path/to/labels/all_labels.csv \
    --results_dir results/
```

### Running the Unit Tests

```bash
pip install pytest
pytest tests/test_truesignal.py -v
```

The test suite covers the MAS algorithm correctness, the incongruence labeling rule across all conflict combinations, the MLP model architecture, and the 24-dimensional feature vector construction.

---

## QLoRA Adapter

The fine-tuned QLoRA adapter (`adapter_model.safetensors`) is 161MB and cannot be stored in this repository. Download it from Google Drive:

**Download: https://drive.google.com/drive/folders/1cSFGHWhdKppjXDlBMOecqU3swBgfPJv4?usp=sharing**

After downloading, place the contents at `qlora_adapter_final/` in your working directory. Load with:

```python
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
model = PeftModel.from_pretrained(base_model, "qlora_adapter_final/")
model.eval()
```

---

## The 24-Dimensional Feature Vector

Every clip is represented as a fixed-length 24-dimensional vector. The exact layout is:

| Index | Feature | Source |
|---|---|---|
| 0 to 6 | Face emotion probabilities | DeepFace, mapped to 7 MELD classes |
| 7 to 13 | Voice emotion probabilities | wav2vec2 IEMOCAP, zero-padded to 7 classes |
| 14 to 20 | Text emotion probabilities | RoBERTa j-hartmann, 7 MELD classes |
| 21 | MAS face-voice | cosine_similarity(face_vec, voice_vec) |
| 22 | MAS face-text | cosine_similarity(face_vec, text_vec) |
| 23 | MAS voice-text | cosine_similarity(voice_vec, text_vec) |

The MELD emotion order throughout the codebase is: anger, disgust, fear, joy, neutral, sadness, surprise.

The wav2vec2 IEMOCAP model outputs only four classes (neutral, happy, angry, sad), so indices 8, 9, and 13 are always zero in the voice vector. MAS scores are computed on these full 7-dimensional vectors, meaning voice-based MAS scores reflect only the four-class signal.

---

## Why MLP Scores 0.9981 and QLoRA Scores 0.5907 on the Same Video

The MLP was trained on the same 24-dimensional feature vector it receives at inference time. When it sees near-zero MAS scores (face-voice: 0.0024, voice-text: 0.0085), it applies the discriminative boundary it learned from 9,988 training examples and outputs 0.9981 with high confidence.

The QLoRA model was fine-tuned on raw MELD video and audio clips. At inference time in the pipeline, it receives a text description of the pre-extracted emotion vectors and MAS scores rather than raw media. The gap between what it trained on and what it receives at inference makes it reason more conservatively, landing at 0.5907.

Both models agree on the binary outcome: the video is emotionally incongruent, and the dominant conflict is between face and voice. The score difference reflects their architectural approach, not a disagreement about the underlying result.

---

## Dataset

This project uses MELD (Multimodal EmotionLines Dataset), a collection of 13,715 video clips from the TV show Friends, each labeled with one of seven emotion categories: anger, disgust, fear, joy, neutral, sadness, surprise. It is the only large-scale dataset providing face video, audio, and text simultaneously in naturalistic conversational settings.

MELD does not provide incongruence labels. We constructed them using a rule validated by two independent human raters achieving Cohen's Kappa of 0.82: a clip is labeled incongruent if at least one pair of modalities shows different top-1 emotions and both modalities have confidence at or above 0.60.

| Split | Clips | Incongruent | Congruent |
|---|---|---|---|
| Train | 9,988 | 6,799 (68.1%) | 3,189 (31.9%) |
| Dev | 1,112 | 751 (67.5%) | 361 (32.5%) |
| Test | 2,615 | 1,769 (67.6%) | 846 (32.4%) |

---

## Compute Requirements

| Task | Hardware | Approximate time |
|---|---|---|
| MELD clip extraction | A100 40GB | 8 hours |
| Incongruence label construction | A100 40GB | 8 hours |
| MLP training | A100 40GB | Under 5 minutes |
| QLoRA fine-tuning (Unsloth, 3 epochs) | A100 40GB | 45 minutes |
| QLoRA inference on full test set | A100 40GB | 2 hours |
| MLP inference per clip | CPU | Under 1 millisecond |
| Gradio demo, MLP version | Mac CPU | Instant |

---

## Research Questions

**RQ1 — Modal Dominance:** Which modality produces the strongest incongruence detection when used as the sole input?

Face dominates (F1 = 0.822), followed by voice (F1 = 0.814), then text (F1 = 0.796). This confirms the visual bias hypothesis. The face is the hardest channel to consciously control, and the model learned this from data.

**RQ2 — Fine-Tuning Effect:** Does QLoRA fine-tuning produce meaningfully higher accuracy than zero-shot baselines?

QLoRA reaches F1 = 0.9935 versus the MLP baseline of F1 = 0.8997, a gain of 9.38 F1 points. Teaching the model specifically about cross-modal incongruence rather than relying on general knowledge is essential.

**RQ3 — Coaching Relevance:** Do coaching suggestions correctly identify the incongruent modality pair?

Both coaching outputs, from the MLP run and the QLoRA run, correctly named the face-voice conflict, addressed the voice tone mismatch with specific actionable guidance, and maintained an empathetic professional tone. Confirmed for face-voice conflicts. A full multi-participant user study is the natural next step for complete evaluation across all conflict types.

---

## Known Limitations

The MLP fusion model has near-zero F1 on the congruent class in isolation, a direct consequence of the 68/32 class imbalance in training data. The QLoRA model handles both classes nearly perfectly. Future work would apply class-weighted loss to the MLP.

The QLoRA model was fine-tuned on raw MELD media but receives text feature descriptions at inference. Closing this modality gap by fine-tuning directly on the text-formatted feature prompt would likely increase inference confidence.

MELD consists entirely of scripted performances from a TV show. The out-of-distribution test on a real interview video is encouraging, but a larger-scale real-world evaluation is needed before drawing broad conclusions about generalization.

---

## Troubleshooting

**sklearn version error when loading scaler:**
pip install scikit-learn==1.6.1

**SpeechBrain GPU tensor error on Colab:**
Add .detach().cpu().numpy() after out_prob tensor operations

**Gradio event loop conflict when relaunching:**
Call gr.close_all() and sleep(3) before launching new instance

**DeepFace tf-keras missing:**
pip install tf-keras

**ElevenLabs model deprecated error:**
Use eleven_turbo_v2_5 — not eleven_monolingual_v1

**Python version incompatibility:**
Use Python 3.11 via conda — 3.12 and 3.13 break SpeechBrain

**QLoRA response without JSON curly braces:**
Parser handles this automatically with 3-tier fallback

**Colab session expires mid-run:**
All files saved to Google Drive — rerun from any cell safely

## Resources

Find our presentation here: https://docs.google.com/presentation/d/1KBzXSHHOGGxY-chvKvqM9iDrFnUO5gND/edit?usp=sharing

Find our demo recording here: https://drive.google.com/file/d/1v1dG0fEiem4KXqZIDs6Jd4yoA1D89NUH/view?usp=sharing

Find our QLoRA adapter here: https://drive.google.com/drive/folders/1cSFGHWhdKppjXDlBMOecqU3swBgfPJv4?usp=sharing


## Citation

```
Bacham, S.V.T., Yerraguntla, S., and Dusari, S.T. (2026).
TrueSignal: Real-Time Multimodal Emotional Incongruence Detection.
STAT GR5293, Columbia University.
```
