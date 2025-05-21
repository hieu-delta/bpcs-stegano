import os
import cv2
import numpy as np
import json
from scipy.ndimage import label

def calculate_delta(block):
    """Tính độ phức tạp delta cho khối 8x8 theo công thức số vùng đen-trắng."""
    labeled_black, num_black = label(block == 0)  # Vùng đen
    labeled_white, num_white = label(block == 1)  # Vùng trắng
    k = num_black + num_white  # Tổng số vùng
    max_regions = 8 * 8  # 64 (tối đa mỗi pixel là một vùng)
    delta = k / max_regions if max_regions > 0 else 0
    return delta

def count_noise_blocks_improved(image_folder, delta_threshold=0.14):
    """Đếm số khối nhiễu trong tất cả ảnh màu trong thư mục."""
    noise_block_counts = {}  # Lưu số khối nhiễu: {image_name: count}
    
    # Duyệt qua tất cả file trong thư mục
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Không thể đọc ảnh: {filename}")
                continue
            
            height, width, _ = img.shape
            noise_block_counts[filename] = 0
            
            # Tách kênh màu (B, G, R)
            channels = cv2.split(img)  # BGR trong OpenCV
            channel_names = ['Blue', 'Green', 'Red']
            
            for ch_idx, channel in enumerate(channels):
                # Phân tách bit-plane
                bit_planes = [(channel >> i) & 1 for i in range(8)]  # Bit 0 (LSB) đến 7 (MSB)
                
                for plane_idx, bit_plane in enumerate(bit_planes):
                    # Chia bit-plane thành khối 8x8
                    for i in range(0, height, 8):
                        for j in range(0, width, 8):
                            if i + 8 <= height and j + 8 <= width:  # Kiểm tra khối đầy đủ
                                block = bit_plane[i:i+8, j:j+8]
                                delta = calculate_delta(block)
                                if delta >= delta_threshold:
                                    noise_block_counts[filename] += 1
            
            print(f"Đã xử lý ảnh: {filename}, tìm thấy {noise_block_counts[filename]} khối nhiễu")
    
    # Lưu kết quả vào file JSON
    with open('noise_blocks_improved.json', 'w') as f:
        json.dump(noise_block_counts, f, indent=4)
    
    return noise_block_counts

# Thư mục chứa ảnh
image_folder = "../input/frames/"  # Thay bằng đường dẫn thực tế
count_noise_blocks_improved(image_folder, delta_threshold=0.14)