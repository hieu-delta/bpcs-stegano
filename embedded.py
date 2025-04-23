import numpy as np
from PIL import Image
import json
import logging

# Thiết lập logging (đặt DEBUG để xem chi tiết pixel nếu cần)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Để bật debug: logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Hàm chuyển chuỗi thành nhị phân và chia thành khối 8x8
def text_to_binary_blocks(message, block_size=8):
    try:
        binary = ''.join(format(ord(c), '08b') for c in message)
        while len(binary) % (block_size * block_size) != 0:
            binary += '0'
        binary_blocks = []
        for i in range(0, len(binary), block_size * block_size):
            block_bits = binary[i:i + block_size * block_size]
            block = np.array([int(b) for b in block_bits], dtype=np.uint8).reshape(block_size, block_size)
            binary_blocks.append(block)
        logging.info(f"Converted message to {len(binary_blocks)} binary blocks")
        return binary_blocks
    except Exception as e:
        logging.error(f"Failed to convert message to binary blocks: {str(e)}")
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
        logging.info("Successfully extracted bit planes")
        return bit_planes
    except Exception as e:
        logging.error(f"Failed to extract bit planes: {str(e)}")
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

# Hàm conjugate khối
def conjugate_block(block):
    try:
        checkerboard = np.array([[0, 1] * 4, [1, 0] * 4] * 4, dtype=np.uint8)
        conjugated = block ^ checkerboard
        return conjugated
    except Exception as e:
        logging.error(f"Failed to conjugate block: {str(e)}")
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

# Hàm giấu tin (thay thế toàn bộ khối nhiễu)
def hide_message(image, binary_blocks, noise_blocks, frame_index, block_size=8, threshold=0.3, start_block_index=0):
    try:
        image_array = np.array(image, dtype=np.uint8)
        height, width, channels = image_array.shape
        block_index = start_block_index
        location_map = []
        
        # Kiểm tra số lượng khối nhiễu khả dụng
        total_noise_blocks = sum(len(blocks) for channel in noise_blocks for blocks in channel)
        if total_noise_blocks < len(binary_blocks) - start_block_index:
            logging.error(f"Not enough noise blocks ({total_noise_blocks}) for {len(binary_blocks) - start_block_index} binary blocks in frame {frame_index}")
            raise ValueError("Insufficient noise blocks")
        logging.info(f"Frame {frame_index} has {total_noise_blocks} noise blocks, need {len(binary_blocks) - start_block_index}")
        
        for channel in range(channels):
            for plane_idx in range(8):
                for i, j in noise_blocks[channel][plane_idx]:
                    if block_index >= len(binary_blocks):
                        break
                    block = binary_blocks[block_index]
                    alpha = calculate_complexity(block)
                    conjugated = False
                    if alpha <= threshold:
                        block = conjugate_block(block)
                        conjugated = True
                    # Thay thế khối nhiễu
                    for bi in range(block_size):
                        for bj in range(block_size):
                            if i + bi >= height or j + bj >= width:
                                logging.error(f"Out of bounds access at ({i+bi}, {j+bj}) in frame {frame_index}")
                                raise ValueError("Invalid pixel coordinates")
                            pixel = image_array[i + bi, j + bj, channel]
                            # Thay vì dùng |= và &=, đặt/xóa bit rõ ràng
                            if block[bi, bj] == 1:
                                new_pixel = pixel | (1 << plane_idx)  # Đặt bit
                            else:
                                new_pixel = pixel & ~(1 << plane_idx)  # Xóa bit
                            new_pixel = new_pixel & 0xFF  # Đảm bảo uint8
                            if new_pixel < 0 or new_pixel > 255:
                                logging.error(f"Invalid pixel value {new_pixel} at ({i+bi}, {j+bj}, channel={channel}, plane={plane_idx})")
                                raise ValueError(f"Pixel value {new_pixel} out of bounds")
                            logging.debug(f"Pixel at ({i+bi}, {j+bj}, channel={channel}, plane={plane_idx}): {pixel} -> {new_pixel}")
                            image_array[i + bi, j + bj, channel] = new_pixel
                    location_map.append((frame_index, channel, plane_idx, i, j, conjugated))
                    block_index += 1
                if block_index >= len(binary_blocks):
                    break
            if block_index >= len(binary_blocks):
                break
        
        logging.info(f"Embedded {block_index - start_block_index} blocks in frame {frame_index}")
        return Image.fromarray(image_array), location_map, block_index
    except Exception as e:
        logging.error(f"Failed to embed message in frame {frame_index}: {str(e)}")
        raise

# Hàm lưu thông tin vào key_K
def update_key_K(key_K, noise_blocks, location_map, stego_frame_paths, output_path="key_K.json"):
    try:
        key_K["bit_planes"] = [[len(nb) for nb in channel] for channel in noise_blocks]
        key_K["noise_blocks"] = noise_blocks
        key_K["location_map"] = location_map
        key_K["stego_frame_paths"] = stego_frame_paths
        with open(output_path, "w") as f:
            json.dump(key_K, f)
        logging.info(f"Successfully updated key_K to {output_path}")
    except Exception as e:
        logging.error(f"Failed to update key_K: {str(e)}")
        raise

# Thực thi
try:
    key_K = load_key_K()
    selected_frame_indices = key_K["selected_frame_indices"]
    frame_paths = key_K["frame_paths"]
    message = read_message("plain.txt")
    binary_blocks = text_to_binary_blocks(message)
    
    location_map = []
    stego_frame_paths = frame_paths.copy()
    block_index = 0
    noise_blocks_all = []
    
    for frame_index in selected_frame_indices:
        frame_path = frame_paths[frame_index]
        frame = Image.open(frame_path)
        bit_planes = get_bit_planes(frame)
        noise_blocks = [[find_noise_blocks(bit_planes[c][i]) for i in range(8)] for c in range(3)]
        noise_blocks_all.append(noise_blocks)
        
        stego_image, frame_location_map, new_block_index = hide_message(
            frame, binary_blocks, noise_blocks, frame_index, start_block_index=block_index
        )
        stego_frame_path = f"frames/stego_frame_{frame_index}.png"
        stego_image.save(stego_frame_path)
        stego_frame_paths[frame_index] = stego_frame_path
        location_map.extend(frame_location_map)
        block_index = new_block_index
        if block_index >= len(binary_blocks):
            break
    
    update_key_K(key_K, noise_blocks_all, location_map, stego_frame_paths)
except Exception as e:
    logging.error(f"Execution failed: {str(e)}")