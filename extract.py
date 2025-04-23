import cv2
import json
import numpy as np
from PIL import Image
import os
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hàm đọc key_K
def load_key_K(input_path="key_K.json"):
    try:
        with open(input_path, "r") as f:
            key_K = json.load(f)
        logging.info(f"Successfully loaded key_K from {input_path}")
        return key_K
    except Exception as e:
        logging.error(f"Failed to load key_K: {str(e)}")
        raise

# Hàm tách khung hình từ video
def extract_frame_from_video(video_path, frame_index, output_dir="temp_frames"):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Cannot extract frame {frame_index}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_img = Image.fromarray(frame_rgb)
        frame_path = os.path.join(output_dir, f"extracted_frame_{frame_index}.png")
        frame_img.save(frame_path)
        cap.release()
        logging.info(f"Successfully extracted frame {frame_index} to {frame_path}")
        return frame_path
    except Exception as e:
        logging.error(f"Failed to extract frame {frame_index}: {str(e)}")
        raise

# Hàm conjugate khối
def conjugate_block(block):
    try:
        checkerboard = np.array([[0, 1] * 4, [1, 0] * 4] * 4, dtype=np.uint8)
        conjugated = block ^ checkerboard
        return conjugated
    except Exception as e:
        logging.error(f"Failed to conjugate block: {str(e)}")
        raise

# Hàm tách tin
def extract_message(stego_images, location_map, block_size=8):
    try:
        binary_message = ''
        
        for frame_index, channel, plane_idx, i, j, conjugated in location_map:
            if frame_index not in stego_images:
                logging.error(f"Frame {frame_index} not found in stego_images")
                raise ValueError(f"Missing frame {frame_index}")
            stego_image = stego_images[frame_index]
            image_array = np.array(stego_image)
            block = (image_array[i:i+block_size, j:j+block_size, channel] >> plane_idx) & 1
            if conjugated:
                block = conjugate_block(block)
            binary_message += ''.join(str(bit) for bit in block.flatten())
        
        # Chuyển binary thành chuỗi, bỏ qua padding
        message = ''
        for i in range(0, len(binary_message), 8):
            byte = binary_message[i:i+8]
            if len(byte) == 8 and byte != '00000000':
                char = chr(int(byte, 2))
                if 32 <= ord(char) <= 126:
                    message += char
                else:
                    break
        
        logging.info(f"Extracted message: {message}")
        return message
    except Exception as e:
        logging.error(f"Failed to extract message: {str(e)}")
        raise

# Thực thi
try:
    key_K = load_key_K()
    video_path = "stego_video.mp4"
    selected_frame_indices = key_K["selected_frame_indices"]
    location_map = key_K["location_map"]
    
    # Tách các khung hình chứa tin
    stego_images = {}
    for frame_index in selected_frame_indices:
        stego_frame_path = extract_frame_from_video(video_path, frame_index)
        stego_images[frame_index] = Image.open(stego_frame_path)
    
    # Tách tin
    extracted_message = extract_message(stego_images, location_map)
    print("Extracted message:", extracted_message)
except Exception as e:
    logging.error(f"Execution failed: {str(e)}")