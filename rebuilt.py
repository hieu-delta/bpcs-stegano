import json
import logging
import ffmpeg
from PIL import Image
import numpy as np

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

# Hàm lấy thông số video gốc
def get_video_info(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if not video_stream:
            raise ValueError("No video stream found")
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = eval(video_stream['r_frame_rate'])
        bitrate = video_stream.get('bit_rate', '4000k')
        codec = video_stream.get('codec_name', 'h264')
        logging.info(f"Video info: {width}x{height}, {fps} fps, {bitrate} bitrate, {codec} codec")
        return width, height, fps, bitrate, codec
    except Exception as e:
        logging.error(f"Failed to get video info: {str(e)}")
        raise

# Hàm kiểm tra tính toàn vẹn khung hình
def verify_frame_integrity(original_path, extracted_frame):
    try:
        original = np.array(Image.open(original_path))
        difference = np.abs(original.astype(np.int32) - extracted_frame.astype(np.int32))
        max_diff = np.max(difference)
        if max_diff > 0:
            logging.warning(f"Frame integrity check failed: Max pixel difference = {max_diff}")
        else:
            logging.info("Frame integrity check passed")
        return max_diff == 0
    except Exception as e:
        logging.error(f"Failed to verify frame integrity: {str(e)}")
        raise

# Hàm ghép các khung hình thành video MP4
def create_video(stego_frame_paths, input_video="test.mp4", output_path="stego_video.mp4"):
    try:
        # Lấy thông số video gốc
        width, height, fps, bitrate, codec = get_video_info(input_video)
        
        # Tạo video từ các khung hình
        stream = ffmpeg.input('pipe:', format='image2pipe', pix_fmt='rgb24', vcodec='png', framerate=fps)
        stream = ffmpeg.output(
            stream, output_path, vcodec='libx264', pix_fmt='yuv420p',
            video_bitrate=bitrate, r=fps, s=f'{width}x{height}',
            preset='veryslow', crf=18
        )
        process = ffmpeg.run_async(stream, pipe=True)
        
        # Gửi các khung hình qua pipe
        for frame_path in stego_frame_paths:
            frame = Image.open(frame_path)
            frame_array = np.array(frame)
            process.stdin.write(frame_array.tobytes())
        
        process.stdin.close()
        process.wait()
        
        # Kiểm tra tính toàn vẹn khung hình
        import cv2
        cap = cv2.VideoCapture(output_path)
        for i, frame_path in enumerate(stego_frame_paths):
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Failed to read frame {i} from video")
                raise ValueError("Video frame read error")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            verify_frame_integrity(frame_path, frame_rgb)
        cap.release()
        
        logging.info(f"Successfully created video at {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to create video: {str(e)}")
        raise

# Thực thi
try:
    key_K = load_key_K()
    stego_frame_paths = key_K["stego_frame_paths"]
    output_video = create_video(stego_frame_paths)
except Exception as e:
    logging.error(f"Execution failed: {str(e)}")