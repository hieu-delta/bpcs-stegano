import cv2
import os
import json
from PIL import Image
import numpy as np
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hàm tính số block bit cần thiết từ thông điệp
def calculate_required_blocks(message, block_size=8):
    try:
        binary_length = len(message) * 8  # Mỗi ký tự 8 bit
        block_bits = block_size * block_size  # 64 bit/block
        num_blocks = (binary_length + block_bits - 1) // block_bits  # Làm tròn lên
        logging.info(f"Message '{message}' requires {num_blocks} blocks")
        return num_blocks
    except Exception as e:
        logging.error(f"Failed to calculate required blocks: {str(e)}")
        raise

# Hàm đọc thông điệp từ plain.txt
def read_message(file_path="plain.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            message = f.read().strip()
        if not message:
            raise ValueError("Empty message in plain.txt")
        logging.info(f"Read message from {file_path}: {message}")
        return message
    except Exception as e:
        logging.error(f"Failed to read message from {file_path}: {str(e)}")
        raise

# Hàm tính độ phức tạp của một khối 8x8
def calculate_complexity(block):
    try:
        k = 0
        for i in range(8):
            for j in range(7):
                if block[i, j] != block[i, j + 1]:
                    k += 1
        for j in range(8):
            for i in range(7):
                if block[i, j] != block[i + 1, j]:
                    k += 1
        n = 8
        alpha = k / (2 * 2 * n * (n - 1))
        return alpha
    except Exception as e:
        logging.error(f"Failed to calculate complexity: {str(e)}")
        raise

# Hàm tìm các khối nhiễu
def find_noise_blocks(bit_plane, block_size=8, threshold=0.3):
    try:
        height, width = bit_plane.shape
        noise_blocks = []
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                if i + block_size <= height and j + block_size <= width:
                    block = bit_plane[i:i+block_size, j:j+block_size]
                    alpha = calculate_complexity(block)
                    if alpha > threshold:
                        noise_blocks.append((i, j))
        return noise_blocks
    except Exception as e:
        logging.error(f"Failed to find noise blocks: {str(e)}")
        raise

# Hàm chia ảnh thành các mặt phẳng bit (RGB)
def get_bit_planes(image):
    try:
        image_array = np.array(image, dtype=np.uint8)
        height, width, channels = image_array.shape
        bit_planes = np.zeros((channels, 8, height, width), dtype=np.uint8)
        for channel in range(channels):
            for bit in range(8):
                bit_planes[channel, bit] = (image_array[:, :, channel] >> bit) & 1
        return bit_planes
    except Exception as e:
        logging.error(f"Failed to extract bit planes: {str(e)}")
        raise

# Hàm phân tích số lượng khối nhiễu của một khung hình
def analyze_frame_noise_blocks(frame_path, block_size=8, threshold=0.3):
    try:
        frame = Image.open(frame_path)
        bit_planes = get_bit_planes(frame)
        total_noise_blocks = 0
        for channel in range(3):
            for bit in range(8):
                noise_blocks = find_noise_blocks(bit_planes[channel][bit], block_size, threshold)
                total_noise_blocks += len(noise_blocks)
        logging.info(f"Frame {frame_path}: {total_noise_blocks} noise blocks")
        return total_noise_blocks
    except Exception as e:
        logging.error(f"Failed to analyze frame {frame_path}: {str(e)}")
        return 0

# Hàm tách video thành các khung hình RGB
def extract_frames(video_path, output_dir="frames"):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_img = Image.fromarray(frame_rgb)
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.png")
            frame_img.save(frame_path)
            frames.append(frame_path)
            frame_count += 1
        cap.release()
        logging.info(f"Successfully extracted {frame_count} frames to {output_dir}")
        return frames
    except Exception as e:
        logging.error(f"Failed to extract frames: {str(e)}")
        raise

# Hàm chọn khung hình dựa trên số lượng khối nhiễu
def select_frames(frames, num_blocks_needed):
    try:
        noise_counts = [(i, analyze_frame_noise_blocks(frame)) for i, frame in enumerate(frames)]
        noise_counts.sort(key=lambda x: x[1], reverse=True)
        
        selected_indices = []
        total_noise_blocks = 0
        for index, count in noise_counts:
            selected_indices.append(index)
            total_noise_blocks += count
            if total_noise_blocks >= num_blocks_needed:
                break
        
        if total_noise_blocks < num_blocks_needed:
            logging.error(f"Not enough noise blocks ({total_noise_blocks}) for {num_blocks_needed} blocks")
            raise ValueError("Insufficient noise blocks across all frames")
        
        logging.info(f"Selected frames {selected_indices} with {total_noise_blocks} noise blocks for {num_blocks_needed} blocks")
        return selected_indices
    except Exception as e:
        logging.error(f"Failed to select frames: {str(e)}")
        raise

# Hàm lưu thông tin khung hình vào khóa K
def save_key_K(frames, selected_frame_indices, output_path="key_K.json"):
    try:
        key_K = {
            "selected_frame_indices": selected_frame_indices,
            "frame_paths": frames
        }
        with open(output_path, "w") as f:
            json.dump(key_K, f)
        logging.info(f"Successfully saved key_K to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save key_K: {str(e)}")
        raise

# Thực thi
try:
    video_path = "test.mp4"
    frames = extract_frames(video_path)
    message = read_message("plain.txt")
    num_blocks_needed = calculate_required_blocks(message)
    selected_frame_indices = select_frames(frames, num_blocks_needed)
    save_key_K(frames, selected_frame_indices)
except Exception as e:
    logging.error(f"Execution failed: {str(e)}")