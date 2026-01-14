import os
from safetensors.torch import load_file, save_file

# --- CẤU HÌNH ---
# Dẫn tới file checkpoint bạn muốn test (ví dụ ở bước 2000)
checkpoint_path = "chatterbox_output/model.safetensors" 

# Tên file sạch sẽ để dùng chạy inference
output_path = "chatterbox_output/t3_finetuned_step2000.safetensors" 
# ----------------
print(f"Đang đọc checkpoint: {checkpoint_path}")
try:
    state_dict = load_file(checkpoint_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        # Code train bọc model trong biến 't3', nên tên layer bị dính tiền tố 't3.'
        # Cần cắt bỏ chữ 't3.' đi thì inference mới hiểu
        if key.startswith("t3."):
            new_key = key[3:]  # Bỏ 3 ký tự đầu
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
            
    print(f"Đang lưu file đã sửa: {output_path}")
    save_file(new_state_dict, output_path)
    print(" Xong! Bạn có thể dùng file này để chạy inference.")

except Exception as e:
    print(f" Lỗi: {e}")
    print("Bạn kiểm tra lại đường dẫn file model.safetensors nhé.")