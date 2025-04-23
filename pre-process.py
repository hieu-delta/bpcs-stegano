import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

# Hàm đọc key_K
def load_key_K(input_path="key_K.json"):
    with open(input_path, "r") as f:
        return json.load(f)

# Hàm chia ảnh thành các mặt phẳng bit
def get_bit_planes(image):
    image_array = np.array(image)
    height, width = image_array.shape
    bit_planes = np.zeros((8, height, width), dtype=np.uint8)
    
    for bit in range(8):
        bit_planes[bit] = (image_array >> bit) & 1
    return bit_planes

# Hàm tính độ phức tạp của một khối 8x8
def calculate_complexity(block):
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

# Hàm tìm các khối nhiễu
def find_noise_blocks(bit_plane, block_size=8, threshold=0.3):
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

# Hàm lưu thông tin khối nhiễu vào key_K
def update_key_K(key_K, noise_blocks, output_path="key_K.json"):
    key_K["bit_planes"] = [len(nb) for nb in noise_blocks]
    key_K["noise_blocks"] = noise_blocks
    with open(output_path, "w") as f:
        json.dump(key_K, f)

# Thực thi
key_K = load_key_K()
selected_frame_path = key_K["frame_paths"][key_K["selected_frame_index"]]
selected_frame = Image.open(selected_frame_path).convert("L")
bit_planes = get_bit_planes(selected_frame)
noise_blocks = [find_noise_blocks(bit_planes[i]) for i in range(8)]
update_key_K(key_K, noise_blocks)

# Hiển thị một mặt phẳng bit
plt.imshow(bit_planes[0], cmap='gray')
plt.title('Bit Plane 0')
plt.savefig('bit_plane_0.png')