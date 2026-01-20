import os
import csv
import soundfile as sf
import librosa
from tqdm import tqdm
from faster_whisper import WhisperModel

# =====================
# CONFIG
# =====================
INPUT_WAV_DIR = "/kaggle/working/ft_voice/voice_v2/clean_dataset_nu"
OUTPUT_DIR = "/kaggle/working/ft_voice/voice_v2/data/v2/output_nu"
CLIP_DIR = os.path.join(OUTPUT_DIR, "clips")

TARGET_SR = 16000
LANGUAGE = "vi"

DEVICE = "cuda"       # "cuda" hoặc "cpu"
COMPUTE_TYPE = "int8" # cuda: int8 / float16 | cpu: int8

MODEL_SIZE = "large-v3"

# =====================
# SETUP
# =====================
os.makedirs(CLIP_DIR, exist_ok=True)

print("🔄 Loading faster-whisper model...")
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

wav_files = sorted(
    f for f in os.listdir(INPUT_WAV_DIR)
    if f.lower().endswith(".wav")
)

metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")

# =====================
# PROCESS
# =====================
total_clip = 0

with open(metadata_path, "w", encoding="utf-8", newline="") as meta_f:
    writer = csv.writer(meta_f, delimiter="|")

    for wav_name in tqdm(wav_files, desc="Processing wav files"):
        wav_path = os.path.join(INPUT_WAV_DIR, wav_name)
        base_name = os.path.splitext(wav_name)[0]

        try:
            audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        except Exception as e:
            print(f"❌ Load lỗi {wav_name}: {e}")
            continue

        segments, info = model.transcribe(
            wav_path,
            language=LANGUAGE,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=1000
            )
        )

        stt = 0
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue

            start_ms = int(seg.start * 1000)
            end_ms = int(seg.end * 1000)

            if end_ms - start_ms < 1000:
                continue

            start_sample = int(seg.start * TARGET_SR)
            end_sample = int(seg.end * TARGET_SR)
            clip_audio = audio[start_sample:end_sample]

            if len(clip_audio) < TARGET_SR:
                continue

            clip_name = f"{base_name}_{start_ms}-{end_ms}_{stt:04d}.wav"
            clip_path = os.path.join(CLIP_DIR, clip_name)

            sf.write(clip_path, clip_audio, TARGET_SR)

            # ===== LOG =====
            print("\n------------------------------")
            print(f"[FILE] {clip_name}")
            print(f"[TEXT] {text}")

            writer.writerow([clip_name, text])

            stt += 1
            total_clip += 1

print("\n✅ HOÀN TẤT")
print(f"🎧 Tổng clip: {total_clip}")
print("📁 Clips   :", CLIP_DIR)
print("📄 Metadata:", metadata_path)
