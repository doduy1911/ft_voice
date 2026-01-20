import json
import os

# Đường dẫn đến file metadata.csv của bạn
METADATA_PATH = "MyTTSDataset/metadata.csv" 
# Hoặc file dataset của bạn

def create_vietnamese_tokenizer():
    # 1. Các ký tự cơ bản của Chatterbox (giữ lại để tương thích)
    chars = [
        " ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", 
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", 
        "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~"
    ]
    
    # 2. Quét toàn bộ dataset để tìm ký tự tiếng Việt
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    found_chars = set()
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) >= 2:
            text = parts[1] # Cột chứa text gốc (hoặc normalized text)
            for char in text:
                if char not in chars:
                    found_chars.add(char)
    
    # Sắp xếp và thêm vào danh sách
    sorted_vietnamese_chars = sorted(list(found_chars))
    chars.extend(sorted_vietnamese_chars)
    
    # 3. Tạo mapping JSON
    vocab_map = {char: idx for idx, char in enumerate(chars)}
    
    # 4. Lưu file (đè lên file cũ hoặc lưu mới)
    save_path = "pretrained_models/tokenizer.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab_map, f, ensure_ascii=False, indent=2)
        
    print(f"Đã tạo tokenizer mới tại {save_path}")
    print(f"Tổng số token (NEW_VOCAB_SIZE): {len(vocab_map)}")
    print("HÃY NHỚ CON SỐ NÀY ĐỂ ĐIỀN VÀO CONFIG!")

if __name__ == "__main__":
    create_vietnamese_tokenizer()