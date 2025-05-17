import os
import cv2
import numpy as np
import json

def conjugate_block(block):
    """Conjugate khối bằng cách XOR với khối cờ vua W_c."""
    W_c = np.array([[1 if (i + j) % 2 == 0 else 0 for j in range(8)] for i in range(8)], dtype=np.uint8)
    conjugated = np.bitwise_xor(block, W_c)
    return conjugated

def bits_to_text(bits, text_length):
    """Chuyển chuỗi bit thành văn bản, sử dụng text_length byte, bỏ qua 32-bit length prefix."""
    # Bỏ qua 32-bit length prefix
    if len(bits) < 32:
        raise ValueError("Chuỗi bit không đủ để chứa độ dài văn bản.")
    text_bits = bits[32:]  # Bắt đầu từ bit thứ 32
    
    # Lấy số byte theo text_length
    byte_count = text_length
    if len(text_bits) < byte_count * 8:
        raise ValueError(f"Chuỗi bit ({len(text_bits)} bit) không đủ để giải mã {byte_count} byte.")
    
    # Chuyển từng nhóm 8 bit thành byte
    text_bytes = bytearray()
    for i in range(0, byte_count * 8, 8):
        byte_bits = text_bits[i:i+8]
        if len(byte_bits) == 8:
            byte = int(byte_bits, 2)
            text_bytes.append(byte)
    
    # Gỡ lỗi: In ra vài byte đầu tiên
    print(f"First 10 bytes (hex): {[hex(b) for b in text_bytes[:10]]}")
    
    # Giải mã UTF-8
    try:
        return text_bytes.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("Không thể giải mã chuỗi bit thành văn bản UTF-8 hợp lệ.")

def extract_text(secret_key_file="result/secret_key_traditional.json", stego_folder="result/traditional_stego_frames/"):
    """Tách văn bản từ các ảnh chứa tin dựa trên khóa bí mật."""
    # Đọc khóa bí mật
    try:
        with open(secret_key_file, 'r') as f:
            secret_key = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file {secret_key_file}")
    
    text_length = secret_key.get('text_length', 0)
    blocks = secret_key.get('blocks', {})
    
    # Lấy danh sách tên file ảnh từ secret_key.json
    image_files = list(blocks.keys())
    
    # Kiểm tra ảnh tồn tại
    valid_images = []
    for img_file in image_files:
        img_path = os.path.join(stego_folder, img_file)
        if os.path.exists(img_path):
            valid_images.append((img_path, img_file))
        else:
            print(f"Ảnh {img_file} không tồn tại trong {stego_folder}, bỏ qua.")
    
    if not valid_images:
        raise ValueError("Không có ảnh hợp lệ để tách tin.")
    
    # Tách bit từ các ảnh
    extracted_bits = ""
    channel_names = ['Blue', 'Green', 'Red']
    
    for img_path, img_file in valid_images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Không thể đọc ảnh: {img_file}")
            continue
        
        # Tách kênh
        channels = cv2.split(img)
        
        # Lấy danh sách khối nhiễu
        for block_info in blocks[img_file]:
            ch_idx = channel_names.index(block_info['channel'])
            plane_idx = block_info['bit_plane']
            row, col = block_info['row'], block_info['col']
            is_conjugated = block_info['conjugated']
            
            # Tách bit-plane
            bit_plane = (channels[ch_idx] >> plane_idx) & 1
            
            # Lấy khối 8x8
            block = bit_plane[row:row+8, col:col+8]
            
            # Reverse conjugate nếu cần
            if is_conjugated:
                block = conjugate_block(block)
            
            # Chuyển khối thành chuỗi bit
            block_bits = ''.join(str(bit) for bit in block.flatten())
            extracted_bits += block_bits
        
        print(f"Đã tách bit từ {img_file}")
    
    # Gỡ lỗi: In ra số bit và vài bit đầu tiên
    print(f"Extracted bits: {len(extracted_bits)} bits")
    print(f"First 32 bits (length prefix): {extracted_bits[:32]}")
    
    # Giải mã văn bản
    if not extracted_bits:
        raise ValueError("Không tách được bit nào từ các ảnh.")
    
    # Giải mã văn bản
    text = bits_to_text(extracted_bits, text_length)
    
    # Lưu văn bản tách ra
    with open('./result/extracted_text.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Văn bản đã được tách và lưu vào ./result/extracted_text.txt")

    return text

if __name__ == "__main__":
    extract_text()