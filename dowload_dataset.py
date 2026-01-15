import os
import csv
import io
import soundfile as sf
import librosa
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv() 
# --- CẤU HÌNH ---
DATASET_NAME = "pnnbao-ump/VieNeu-TTS-140h"
OUTPUT_DIR = "MyTTSDataset"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")
MAX_SAMPLES = 2000
HF_TOKEN = os.getenv("HF_TOKEN")   
def main():
    print(f"Dang tai dataset: {DATASET_NAME}...")
    
    try:
        # Load dataset với streaming=True
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True, token=HF_TOKEN)
    except Exception as e:
        print(f"Lỗi khởi tạo dataset: {e}")
        return

    os.makedirs(WAV_DIR, exist_ok=True)
    print("Dang xu ly va luu file (Da sua loi 'array')...")
    
    with open(METADATA_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        
        count = 0
        for item in tqdm(dataset):
            if MAX_SAMPLES and count >= MAX_SAMPLES:
                break
                
            try:
                # 1. Lấy Text
                text_content = item.get('text') or item.get('sentence') or item.get('transcript')
                if not text_content:
                    continue

                # 2. Xử lý Audio (Phần quan trọng đã sửa)
                audio_info = item.get('audio')
                if not audio_info:
                    continue

                audio_array = None
                sr = None

                # Trường hợp A: Đã có sẵn array (Lý tưởng)
                if 'array' in audio_info:
                    audio_array = audio_info['array']
                    sr = audio_info['sampling_rate']
                
                # Trường hợp B: Chỉ có bytes (Lỗi bạn đang gặp) -> Tự decode
                elif 'bytes' in audio_info:
                    audio_bytes = audio_info['bytes']
                    # Dùng soundfile để đọc từ bytes
                    audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                
                # Trường hợp C: Chỉ có path (Ít gặp khi streaming)
                elif 'path' in audio_info:
                     # Nếu cần thiết mới tải, nhưng thường 'bytes' sẽ đi kèm
                     print(f"Bỏ qua mẫu {count}: Chỉ có path, không có data.")
                     continue

                if audio_array is None:
                    continue

                # 3. Resample về 16kHz (Chuẩn cho Chatterbox)
                # Dataset gốc là 24kHz, ta nên đưa về 16kHz để nhẹ và đúng chuẩn train
                if sr != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                    sr = 16000

                # 4. Lưu file
                filename = f"audio_{count:05d}"
                wav_path = os.path.join(WAV_DIR, f"{filename}.wav")
                sf.write(wav_path, audio_array, sr)

                # 5. Ghi Metadata
                writer.writerow([filename, text_content, text_content])
                
                count += 1
                
            except Exception as e:
                # In chi tiết lỗi để debug nếu còn bị
                print(f"\nLỗi mẫu {count}: {e}")
                # Kiểm tra xem key thực tế là gì
                if 'audio' in item:
                    print(f"Keys trong audio: {item['audio'].keys()}")
                continue

    print(f"\n✅ Hoàn tất! Đã lưu {count} mẫu vào '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()