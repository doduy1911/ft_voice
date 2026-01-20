import os
import subprocess
from pydub import AudioSegment, silence

# --- CẤU HÌNH ---
INPUT_FOLDER = "/ft_voice/voice_v2/data_ytb/nu1"      # Folder chứa voice gốc của client
OUTPUT_FOLDER = "/ft_voice/voice_v2/clean_dataset_nu" # Folder chứa kết quả
TEMP_FOLDER = "/ft_voice/voice_v2/temp_denoised_nu"   # Folder trung gian

# Tạo folder nếu chưa có
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

def denoise_audio(input_path, output_path):
    """
    Sử dụng DeepFilterNet qua command line để khử noise.
    DeepFilterNet giữ chất giọng thực tốt hơn các bộ lọc số truyền thống.
    """
    print(f"[*] Đang khử noise: {input_path}...")
    
    # Lệnh chạy DeepFilterNet (deepFilter). 
    # Output mặc định của nó sẽ thêm hậu tố, ta cần xử lý file output sau đó.
    cmd = [
        "deepFilter", 
        input_path, 
        "-o", os.path.dirname(output_path) # Output ra folder
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        
        # DeepFilterNet thường tạo file dạng 'tenfile_DeepFilterNet3.wav'
        # Ta cần rename lại cho đúng flow
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        # Tìm file vừa tạo ra (giả sử dùng model mặc định DeepFilterNet3)
        generated_file = None
        for file in os.listdir(os.path.dirname(output_path)):
            if file.startswith(base_name) and "DeepFilterNet" in file:
                generated_file = os.path.join(os.path.dirname(output_path), file)
                break
        
        if generated_file:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(generated_file, output_path)
            return True
        else:
            print(f"[!] Không tìm thấy file output từ DeepFilter cho {input_path}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"[!] Lỗi khi chạy DeepFilterNet: {e}")
        return False

def trim_silence_and_export(input_path, final_output_path):
    """
    Cắt bỏ khoảng lặng đầu và cuối, chuẩn hóa âm lượng.
    """
    print(f"[*] Đang xử lý khoảng lặng: {input_path}...")
    try:
        audio = AudioSegment.from_wav(input_path)
        
        # 1. Chuẩn hóa âm lượng về -20dBFS (Tiêu chuẩn dataset)
        change_in_dBFS = -20.0 - audio.dBFS
        normalized_sound = audio.apply_gain(change_in_dBFS)
        
        # 2. Detect và trim silence (cắt khoảng lặng > 500ms)
        # Cách đơn giản: Cắt đầu đuôi
        non_silent_audio = silence.split_on_silence(
            normalized_sound,
            min_silence_len=500,
            silence_thresh=-45, # Ngưỡng im lặng (dB)
            keep_silence=200    # Giữ lại 200ms im lặng để giọng không bị cụt
        )
        
        # Ghép lại (nếu file là 1 câu dài bị ngắt quãng) hoặc lấy đoạn dài nhất
        if non_silent_audio:
            output_audio = non_silent_audio[0] 
            for segment in non_silent_audio[1:]:
                output_audio += segment
            
            # Export chuẩn 44100Hz hoặc 22050Hz (tùy model TTS bạn train)
            output_audio.export(final_output_path, format="wav", parameters=["-ar", "44100"])
            print(f"[OK] Đã xong: {final_output_path}")
        else:
            print(f"[!] File {input_path} toàn là im lặng?")
            
    except Exception as e:
        print(f"[!] Lỗi xử lý pydub: {e}")

def main():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.wav', '.mp3', '.m4a'))]
    print(f"Tìm thấy {len(files)} file audio.")

    for file in files:
        raw_path = os.path.join(INPUT_FOLDER, file)
        temp_path = os.path.join(TEMP_FOLDER, file.replace(".mp3", ".wav").replace(".m4a", ".wav"))
        final_path = os.path.join(OUTPUT_FOLDER, file.replace(".mp3", ".wav").replace(".m4a", ".wav"))
        
        # Bước 1: Denoise vào folder Temp
        success = denoise_audio(raw_path, temp_path)
        
        # Bước 2: Cắt gọt và chuẩn hóa vào folder đích
        if success:
            trim_silence_and_export(temp_path, final_path)

    # Dọn dẹp folder temp nếu muốn
    # shutil.rmtree(TEMP_FOLDER) 

if __name__ == "__main__":
    main()