"""
MELD Clip Extraction Pipeline
Extracts frames (OpenCV), audio (ffmpeg), transcripts (Whisper)
for all clips in the MELD dataset.

Usage:
    python pipeline/extract_clips.py \
        --meld_raw  /path/to/meld_raw \
        --meld_proc /path/to/meld_processed \
        --split     all

Requirements: opencv-python-headless, ffmpeg-python, openai-whisper
Hardware:     A100 GPU recommended (~8 hours for full MELD)
"""

import os, cv2, subprocess, json, glob, argparse
import whisper
from tqdm import tqdm


def extract_clip(clip_path: str, out_dir: str, whisper_model) -> dict:
    """
    Extract frames, audio, and transcript from a single MELD clip.

    Args:
        clip_path:     Path to .mp4 file
        out_dir:       Output directory for this clip
        whisper_model: Loaded Whisper model instance

    Returns:
        dict: Metadata with duration, fps, frames list, and transcript

    Raises:
        RuntimeError: If ffmpeg audio extraction fails
    """
    os.makedirs(out_dir, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(clip_path))[0]

    # ── 1. Frame extraction (1 frame per 2 seconds) ───────────────
    cap          = cv2.VideoCapture(clip_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    interval     = max(1, int(fps * 2.0))
    frame_idx, saved_frames = 0, []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            ts  = int(frame_idx / fps)
            dst = f"{out_dir}/frame_{ts}s.jpg"
            cv2.imwrite(dst, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_frames.append(f"frame_{ts}s.jpg")
        frame_idx += 1
    cap.release()

    # Guarantee at least 1 frame for short clips
    if not saved_frames:
        cap = cv2.VideoCapture(clip_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(f"{out_dir}/frame_0s.jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_frames.append("frame_0s.jpg")

    # ── 2. Audio extraction (16kHz mono WAV) ─────────────────────
    audio_path = f"{out_dir}/audio.wav"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", clip_path,
         "-ar", "16000", "-ac", "1", "-f", "wav", audio_path],
        capture_output=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {r.stderr.decode()[:200]}")

    # ── 3. Whisper transcription ──────────────────────────────────
    result     = whisper_model.transcribe(audio_path, language="en",
                                          fp16=True, verbose=False)
    transcript = result["text"].strip()
    with open(f"{out_dir}/transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)

    return {
        "clip_name":    clip_name,
        "duration_sec": round(duration_sec, 2),
        "fps":          round(fps, 2),
        "num_frames":   len(saved_frames),
        "frames":       saved_frames,
        "whisper_text": transcript,
    }


def process_split(split: str, meld_raw: str, meld_proc: str,
                  whisper_model) -> None:
    """Process all clips in a split with resume support (skips done clips)."""
    split_dirs = {
        "train": "train_splits",
        "dev":   "dev_splits_complete",
        "test":  "output_repeated_splits_test",
    }
    clips    = sorted(glob.glob(f"{meld_raw}/{split_dirs[split]}/*.mp4"))
    log_path = f"{meld_proc}/{split}_failed.jsonl"
    processed = skipped = failed = 0

    for clip_path in tqdm(clips, desc=split, unit="clip"):
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        out_dir   = f"{meld_proc}/{split}/{clip_name}"

        # Resume-safe: skip already-extracted clips
        if os.path.exists(f"{out_dir}/transcript.txt"):
            skipped += 1
            continue

        try:
            meta          = extract_clip(clip_path, out_dir, whisper_model)
            meta["split"] = split
            with open(f"{out_dir}/metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
            processed += 1
        except Exception as e:
            failed += 1
            with open(log_path, "a") as f:
                f.write(json.dumps(
                    {"clip": clip_name, "error": str(e)[:300]}) + "\n")

    print(f"{split}: {processed} processed | {skipped} skipped | {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MELD clips")
    parser.add_argument("--meld_raw",  required=True)
    parser.add_argument("--meld_proc", required=True)
    parser.add_argument("--split", default="all",
                        choices=["train", "dev", "test", "all"])
    args = parser.parse_args()

    print("Loading Whisper small...")
    model  = whisper.load_model("small", device="cuda")
    splits = ["train","dev","test"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(split, args.meld_raw, args.meld_proc, model)
