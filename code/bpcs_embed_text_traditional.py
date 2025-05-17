import os
import cv2
import numpy as np
import json
from scipy.ndimage import label

def calculate_delta(block):
    """Tính độ phức tạp delta cho khối 8x8 (dùng để kiểm tra conjugate)."""
    labeled_black, num_black = label(block == 0)
    labeled_white, num_white = label(block == 1)
    k = num_black + num_white
    max_regions = 8 * 8
    delta = k / max_regions if max_regions > 0 else 0
    return delta

def conjugate_block(block):
    """Conjugate khối bằng cách XOR với khối cờ vua W_c."""
    W_c = np.array([[1 if (i + j) % 2 == 0 else 0 for j in range(8)] for i in range(8)], dtype=np.uint8)
    conjugated = np.bitwise_xor(block, W_c)
    return conjugated

def text_to_bits(text):
    """Chuyển văn bản thành chuỗi bit, thêm độ dài ở đầu."""
    text_bytes = text.encode('utf-8')
    length = len(text_bytes)
    length_bits = format(length, '032b')
    text_bits = ''.join(format(byte, '08b') for byte in text_bytes)
    return length_bits + text_bits

def find_noise_blocks(image_path, alpha_threshold=0.3):
    """Tìm chi tiết các khối nhiễu trong ảnh (phương pháp truyền thống)."""
    def calculate_traditional_complexity(block):
        changes = 0
        for i in range(8):
            for j in range(7):
                if block[i, j] != block[i, j+1]:
                    changes += 1
        for j in range(8):
            for i in range(7):
                if block[i, j] != block[i+1, j]:
                    changes += 1
        max_changes = 2 * 8 * (8 - 1)
        alpha = changes / max_changes if max_changes > 0 else 0
        return alpha
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    height, width, _ = img.shape
    noise_blocks = []
    channels = cv2.split(img)
    channel_names = ['Blue', 'Green', 'Red']
    
    for ch_idx, channel in enumerate(channels):
        bit_planes = [(channel >> i) & 1 for i in range(8)]
        for plane_idx, bit_plane in enumerate(bit_planes):
            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    if i + 8 <= height and j + 8 <= width:
                        block = bit_plane[i:i+8, j:j+8]
                        alpha = calculate_traditional_complexity(block)
                        if alpha >= alpha_threshold:
                            noise_blocks.append({
                                'channel': channel_names[ch_idx],
                                'bit_plane': plane_idx,
                                'row': i,
                                'col': j,
                                'alpha': float(alpha)
                            })
    
    return sorted(noise_blocks, key=lambda x: (x['channel'] != 'Blue', x['bit_plane']))

def embed_text(noise_blocks_file="code/noise_blocks_traditional.json", indices_file="input/frame_key.txt"):
    """Nhúng văn bản từ plain.txt vào các ảnh được chọn (phương pháp truyền thống)."""
    try:
        with open(noise_blocks_file, 'r') as f:
            noise_block_counts = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file {noise_blocks_file}")
    
    try:
        with open('input/plain.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError("Không tìm thấy file plain.txt")
    
    bits = text_to_bits(text)
    bits_needed = len(bits)
    
    try:
        with open(indices_file, 'r') as f:
            image_numbers = f.read().strip().split(',')
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file {indices_file}")
    except ValueError:
        raise ValueError(f"Định dạng trong {indices_file} không hợp lệ. Vui lòng cung cấp các số thứ tự ảnh cách nhau bằng dấu phẩy (ví dụ: 1,2,3).")
    
    image_files = [f"frame_{num.strip()}.png" for num in image_numbers if num.strip()]
    
    image_folder = "input/frames/"
    output_folder = "result/traditional_stego_frames/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    valid_images = []
    total_noise_blocks = 0
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        if os.path.exists(img_path) and img_file in noise_block_counts:
            valid_images.append(img_path)
            total_noise_blocks += noise_block_counts[img_file]
        else:
            print(f"Ảnh {img_file} không tồn tại hoặc không có trong {noise_blocks_file}, bỏ qua.")
    
    if not valid_images:
        raise ValueError("Không có ảnh hợp lệ để nhúng tin.")
    
    blocks_needed = (bits_needed + 63) // 64
    if total_noise_blocks < blocks_needed:
        raise ValueError(f"Không đủ dung lượng: Cần {blocks_needed} khối, chỉ có {total_noise_blocks} khối nhiễu.")
    
    noise_blocks_per_image = []
    for img_path in valid_images:
        noise_blocks = find_noise_blocks(img_path)
        noise_blocks_per_image.append(noise_blocks)
    
    bits_per_image = ((bits_needed + len(valid_images) - 1) // len(valid_images) + 63) // 64 * 64
    text_segments = []
    for i in range(0, bits_needed, bits_per_image):
        segment = bits[i:i + bits_per_image]
        if len(segment) < bits_per_image:
            segment = segment + '0' * (bits_per_image - len(segment))
        text_segments.append(segment)
    
    secret_key = {}
    
    for img_idx, (img_path, noise_blocks) in enumerate(zip(valid_images, noise_blocks_per_image)):
        img = cv2.imread(img_path)
        filename = os.path.basename(img_path)
        secret_key[f"{filename}"] = []
        
        channels = list(cv2.split(img))
        channel_names = ['Blue', 'Green', 'Red']
        bit_planes = [[] for _ in range(3)]
        for ch_idx, channel in enumerate(channels):
            bit_planes[ch_idx] = [(channel >> i) & 1 for i in range(8)]
        
        segment_bits = text_segments[img_idx] if img_idx < len(text_segments) else ""
        
        bit_index = 0
        for block_info in noise_blocks:
            if bit_index >= len(segment_bits):
                break
            
            ch_idx = channel_names.index(block_info['channel'])
            plane_idx = block_info['bit_plane']
            row, col = block_info['row'], block_info['col']
            
            block_bits = segment_bits[bit_index:bit_index + 64]
            if len(block_bits) < 64:
                block_bits = block_bits + '0' * (64 - len(block_bits))
            
            block = np.array([int(b) for b in block_bits], dtype=np.uint8).reshape(8, 8)
            
            delta = calculate_delta(block)
            is_conjugated = False
            if delta < 0.14:
                block = conjugate_block(block)
                is_conjugated = True
            
            bit_planes[ch_idx][plane_idx][row:row+8, col:col+8] = block
            secret_key[f"{filename}"].append({
                'channel': block_info['channel'],
                'bit_plane': plane_idx,
                'row': row,
                'col': col,
                'conjugated': is_conjugated
            })
            bit_index += 64
        
        for ch_idx in range(3):
            channel = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for plane_idx in range(8):
                channel |= (bit_planes[ch_idx][plane_idx] << plane_idx)
            channels[ch_idx] = channel
        stego_img = cv2.merge(channels)
        
        output_path = os.path.join(output_folder, f"{filename}")
        cv2.imwrite(output_path, stego_img)
        print(f"Đã nhúng văn bản vào {filename}, lưu tại {output_path}")
    
    with open('result/secret_key_traditional.json', 'w') as f:
        json.dump({'text_length': len(text.encode('utf-8')), 'blocks': secret_key}, f, indent=4)

if __name__ == "__main__":
    embed_text()