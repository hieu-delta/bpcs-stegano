import cv2
import os
import json
from PIL import Image
import numpy as np
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hàm tách video thành các khung hình RGB
def extract_frames(video_path, output_dir="input/frames_2/"):
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

# Thực thi
try:
    video_path = "result/rebuilt_stego_video.mp4"
    frames = extract_frames(video_path)
except Exception as e:
    logging.error(f"Execution failed: {str(e)}")