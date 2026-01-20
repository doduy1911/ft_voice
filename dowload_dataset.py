import os
import csv
import io
import soundfile as sf
import librosa
from datasets import load_dataset, Audio  # <--- Đã thêm Audio để cấu hình
from tqdm import tqdm
from dotenv import load_dotenv

# Load biến môi trường (HF_TOKEN)
load_dotenv() 

# --- CẤU HÌNH ---
DATASET_NAME = "capleaf/viVoice"
OUTPUT_DIR = "MyTTSDataset"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

# Số lượng mẫu muốn tải. (Set None để tải hết)
MAX_SAMPLES = 5000 

HF_TOKEN = os.getenv("HF_TOKEN")   

def main():
    print(f"Đang kết nối tới dataset: {DATASET_NAME}...")
    
    try:
        # 1. Load dataset (Streaming)
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True, token=HF_TOKEN)
        
        # 2. QUAN TRỌNG: Tắt decode tự động
        # Lệnh này bảo thư viện không dùng torchcodec/ffmpeg để giải mã nữa
        # mà trả về raw bytes trực tiếp. Tránh hoàn toàn lỗi ImportError.
        dataset = dataset.cast_column("audio", Audio(decode=False))
        
    except Exception as e:
        print(f"Lỗi khởi tạo dataset: {e}")
        return

    os.makedirs(WAV_DIR, exist_ok=True)
    print(f"Đang xử lý audio và lưu vào {OUTPUT_DIR}...")
    
    with open(METADATA_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        
        count = 0
        # Duyệt qua từng dòng
        for item in tqdm(dataset):
            if MAX_SAMPLES is not None and count >= MAX_SAMPLES:
                print(f"\nĐã đạt giới hạn {MAX_SAMPLES} mẫu. Dừng tải.")
                break
                
            try:
                # 1. Lấy Text
                text_content = item.get('transcript') or item.get('content') or item.get('text') or item.get('sentence')
                if not text_content:
                    continue

                # 2. Xử lý Audio (Lấy từ raw bytes)
                audio_info = item.get('audio')
                if not audio_info:
                    continue

                audio_array = None
                sr = None

                # Vì đã tắt decode=False, dữ liệu sẽ luôn nằm trong 'bytes' hoặc 'path'
                if 'bytes' in audio_info:
                    audio_bytes = audio_info['bytes']
                    if audio_bytes:
                        # Dùng soundfile để đọc trực tiếp từ RAM
                        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                
                # Fallback nếu soundfile không đọc được bytes (hiếm gặp)
                elif 'path' in audio_info:
                     if audio_info['path'].endswith('.wav'):
                         audio_array, sr = sf.read(audio_info['path'])

                if audio_array is None:
                    continue

                # 3. Resample về 16kHz
                if sr != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                    sr = 16000

                # 4. Lưu file WAV
                filename = f"audio_{count:05d}"
                wav_path = os.path.join(WAV_DIR, f"{filename}.wav")
                sf.write(wav_path, audio_array, sr)

                # 5. Ghi Metadata
                writer.writerow([filename, text_content, text_content])
                
                count += 1
                
            except Exception as e:
                # Bỏ qua lỗi nhỏ để không dừng toàn bộ quá trình
                # print(f"Lỗi mẫu {count}: {e}") 
                continue

    print(f"\n✅ Hoàn tất! Đã lưu {count} mẫu vào '{OUTPUT_DIR}'.")
    print(f"👉 File metadata nằm tại: {METADATA_PATH}")

if __name__ == "__main__":
    main()