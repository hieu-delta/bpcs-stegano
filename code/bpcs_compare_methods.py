import os
import cv2
import numpy as np
import json
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error

def calculate_histogram_difference(original_img, stego_img):
    """Tính tổng chênh lệch tuyệt đối của histogram các kênh."""
    diff = 0
    for ch in range(3):
        hist_orig = cv2.calcHist([original_img], [ch], None, [256], [0, 256]).flatten()
        hist_stego = cv2.calcHist([stego_img], [ch], None, [256], [0, 256]).flatten()
        diff += np.sum(np.abs(hist_orig - hist_stego))
    return diff

def compare_methods(
    traditional_noise_file="cnoise_blocks_traditional.json",
    improved_noise_file="noise_blocks_improved.json",
    indices_file="../input/frame_key.txt",
    image_folder="../input/frames/",
    traditional_stego_folder="../result/traditional_stego_frames/",
    improved_stego_folder="../result/stego_frames/",
    text_file="../input/plain.txt"
):
    """So sánh phương pháp BPCS truyền thống và cải tiến."""
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file {text_file}")
    
    text_bytes = len(text.encode('utf-8'))
    bits_needed = text_bytes * 8 + 32
    blocks_needed = (bits_needed + 63) // 64
    
    try:
        with open(indices_file, 'r') as f:
            image_numbers = f.read().strip().split(',')
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file {indices_file}")
    
    image_files = [f"frame_{num.strip()}.png" for num in image_numbers if num.strip()]
    
    try:
        with open(traditional_noise_file, 'r') as f:
            traditional_blocks = json.load(f)
        with open(improved_noise_file, 'r') as f:
            improved_blocks = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Không tìm thấy file: {e}")
    
    traditional_capacity = 0
    improved_capacity = 0
    valid_images = []
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        if os.path.exists(img_path) and img_file in traditional_blocks and img_file in improved_blocks:
            valid_images.append(img_file)
            traditional_capacity += traditional_blocks[img_file]
            improved_capacity += improved_blocks[img_file]
    
    if not valid_images:
        raise ValueError("Không có ảnh hợp lệ để so sánh.")
    
    report = []
    report.append("So sánh phương pháp BPCS truyền thống và cải tiến")
    report.append("=" * 50)
    report.append(f"Văn bản: {text_file} ({text_bytes} bytes, {bits_needed} bits)")
    report.append(f"Số khối cần thiết: {blocks_needed}")
    report.append(f"Danh sách ảnh: {', '.join(valid_images)}")
    report.append("\nDung lượng nhúng (số khối nhiễu):")
    report.append(f"- Truyền thống (α-based): {traditional_capacity} khối")
    report.append(f"- Cải tiến (δ-based): {improved_capacity} khối")
    
    report.append("\nChất lượng ảnh (PSNR, MSE, Histogram Difference):")
    for img_file in valid_images:
        orig_path = os.path.join(image_folder, img_file)
        trad_stego_path = os.path.join(traditional_stego_folder, f"{img_file}")
        impr_stego_path = os.path.join(improved_stego_folder, f"{img_file}")
        
        orig_img = cv2.imread(orig_path)
        trad_stego_img = cv2.imread(trad_stego_path)
        impr_stego_img = cv2.imread(impr_stego_path)
        
        if trad_stego_img is None or impr_stego_img is None:
            report.append(f"\nẢnh {img_file}: Không tìm thấy ảnh stego")
            continue
        
        trad_psnr = peak_signal_noise_ratio(orig_img, trad_stego_img, data_range=255)
        impr_psnr = peak_signal_noise_ratio(orig_img, impr_stego_img, data_range=255)
        trad_mse = mean_squared_error(orig_img, trad_stego_img)
        impr_mse = mean_squared_error(orig_img, impr_stego_img)
        
        trad_hist_diff = calculate_histogram_difference(orig_img, trad_stego_img)
        impr_hist_diff = calculate_histogram_difference(orig_img, impr_stego_img)
        
        report.append(f"\nẢnh {img_file}:")
        report.append(f"- Truyền thống (α-based):")
        report.append(f"  PSNR: {trad_psnr:.2f} dB")
        report.append(f"  MSE: {trad_mse:.2f}")
        report.append(f"  Histogram Difference: {trad_hist_diff:.2f}")
        report.append(f"- Cải tiến (δ-based):")
        report.append(f"  PSNR: {impr_psnr:.2f} dB")
        report.append(f"  MSE: {impr_mse:.2f}")
        report.append(f"  Histogram Difference: {impr_hist_diff:.2f}")
    
    with open('../result/comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print("Báo cáo so sánh đã được lưu vào ../result/comparison_report.txt")

if __name__ == "__main__":
    compare_methods()