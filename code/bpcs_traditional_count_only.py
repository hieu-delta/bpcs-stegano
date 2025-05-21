import os
import cv2
import numpy as np
import json

def calculate_traditional_complexity(block):
    """Tính độ phức tạp alpha theo phương pháp BPCS truyền thống."""
    changes = 0
    # Đếm số lần thay đổi bit trên các hàng
    for i in range(8):
        for j in range(7):
            if block[i, j] != block[i, j+1]:
                changes += 1
    # Đếm số lần thay đổi bit trên các cột
    for j in range(8):
        for i in range(7):
            if block[i, j] != block[i+1, j]:
                changes += 1
    # Chuẩn hóa theo số lần thay đổi tối đa (2 * 8 * (8-1) = 112)
    max_changes = 2 * 8 * (8 - 1)
    alpha = changes / max_changes if max_changes > 0 else 0
    return alpha

def count_noise_blocks(image_folder="../input/frames/", output_file="noise_blocks_traditional.json", alpha_threshold=0.3):
    """Đếm số khối nhiễu cho tất cả ảnh theo phương pháp BPCS truyền thống."""
    noise_block_counts = {}
    
    for img_file in os.listdir(image_folder):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Không thể đọc ảnh: {img_path}")
                continue
            
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
            
            # Sắp xếp ưu tiên kênh Blue và bit-plane thấp
            noise_blocks = sorted(noise_blocks, key=lambda x: (x['channel'] != 'Blue', x['bit_plane']))
            noise_block_counts[img_file] = len(noise_blocks)
            print(f"Đã đếm {len(noise_blocks)} khối nhiễu cho {img_file}")
    
    with open(output_file, 'w') as f:
        json.dump(noise_block_counts, f, indent=4)
    print(f"Kết quả đã được lưu vào {output_file}")

if __name__ == "__main__":
    count_noise_blocks()