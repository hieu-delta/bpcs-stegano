import os
import cv2
import numpy as np
import subprocess
import tempfile
import shutil
import argparse
import json

def get_video_fps(video_path="input/test.mp4"):
    """Lấy fps của video gốc bằng ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "json",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        fps_str = data["streams"][0]["r_frame_rate"]
        num, denom = map(int, fps_str.split("/"))
        return num / denom
    except (subprocess.CalledProcessError, FileNotFoundError, KeyError):
        print(f"Cảnh báo: Không thể lấy fps từ {video_path}. Sử dụng fps mặc định.")
        return 30.0

def read_indices(indices_file="input/frame_key.txt"):
    """Đọc danh sách chỉ số khung hình từ frame_key.txt."""
    try:
        with open(indices_file, 'r') as f:
            indices = [int(num.strip()) for num in f.read().strip().split(',') if num.strip()]
        return sorted(indices)
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy {indices_file}")
    except ValueError:
        raise ValueError(f"Định dạng trong {indices_file} không hợp lệ. Vui lòng cung cấp các số thứ tự cách nhau bằng dấu phẩy (ví dụ: 1,2,3).")

def get_frame_files(stego_folder, original_folder, indices_file="input/frame_key.txt"):
    """
    Lấy danh sách tất cả khung hình: stego frames cho các chỉ số trong indices_file,
    original frames cho các chỉ số còn lại.
    """
    stego_folder = os.path.abspath(stego_folder)
    original_folder = os.path.abspath(original_folder)
    
    # Đọc chỉ số khung hình giấu tin
    stego_indices = read_indices(indices_file)
    
    # Lấy tất cả khung hình gốc từ original_folder
    original_files = sorted(
        [f for f in os.listdir(original_folder) if f.startswith("frame_") and f.endswith(".png")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    if not original_files:
        raise ValueError(f"Không tìm thấy khung hình gốc trong {original_folder}")
    
    total_frames = max(int(f.split("_")[1].split(".")[0]) for f in original_files)
    
    # Kiểm tra khung hình giấu tin
    stego_files = {
        int(f.split("_")[1].split(".")[0]): f
        for f in os.listdir(stego_folder)
        if f.startswith("frame_") and f.endswith(".png")
    }
    
    # Kiểm tra tính hợp lệ của chỉ số
    for idx in stego_indices:
        if idx > total_frames:
            raise ValueError(f"Chỉ số {idx} trong {indices_file} vượt quá số khung hình gốc ({total_frames})")
        if idx not in stego_files:
            raise ValueError(f"Khung hình giấu tin frame_{idx}.png không tồn tại trong {stego_folder}")
    
    # Tạo danh sách khung hình theo thứ tự
    frame_files = []
    for i in range(1, total_frames + 1):
        if i in stego_indices:
            frame_files.append(os.path.join(stego_folder, stego_files[i]))
        else:
            orig_file = f"frame_{i}.png"
            if orig_file in original_files:
                frame_files.append(os.path.join(original_folder, orig_file))
            else:
                raise ValueError(f"Khung hình gốc frame_{i}.png không tồn tại trong {original_folder}")
    
    # Ghi log danh sách khung hình
    print(f"Tổng số khung hình: {len(frame_files)}")
    print("Danh sách khung hình được sử dụng:")
    for i, f in enumerate(frame_files, 1):
        print(f"Khung hình {i}: {f}")
    
    return frame_files, stego_indices

def reconstruct_video_ffmpeg(stego_folder, original_folder, output_video, indices_file="input/frame_key.txt", fps=30, codec="libx264"):
    """
    Khôi phục video sử dụng FFmpeg, kết hợp khung hình giấu tin và khung hình gốc.

    Args:
        stego_folder (str): Thư mục chứa khung hình giấu tin (e.g., result/stego_frames/).
        original_folder (str): Thư mục chứa khung hình gốc (e.g., input/frames/).
        output_video (str): Đường dẫn video đầu ra (e.g., result/stego_video.mp4).
        indices_file (str): File chứa các chỉ số khung hình giấu tin (e.g., input/frame_key.txt).
        fps (float): Tốc độ khung hình.
        codec (str): Codec video (libx264 hoặc ffv1).
    """
    # Tạo thư mục đầu ra
    output_dir = os.path.dirname(output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Lấy danh sách khung hình
    frame_files, stego_indices = get_frame_files(stego_folder, original_folder, indices_file)
    
    if not frame_files:
        raise ValueError("Không có khung hình nào để tạo video. Kiểm tra thư mục khung hình và file chỉ số.")
    
    # Tạo file danh sách khung hình
    temp_list = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    for frame_file in frame_files:
        temp_list.write(f"file '{frame_file}'\n")
    temp_list.close()
    
    # Lệnh FFmpeg
    ffmpeg_cmd = [
        "ffmpeg", "-y",  # Ghi đè file đầu ra
        "-f", "concat",  # Đọc danh sách khung hình
        "-safe", "0",  # Cho phép đường dẫn tuyệt đối
        "-i", temp_list.name,  # File danh sách khung hình
    ]
    
    if codec == "libx264":
        ffmpeg_cmd.extend(["-c:v", "libx264", "-crf", "0", "-preset", "fast"])  # Lossless H.264
        pix_fmt = "yuv444p"  # Planar YUV, lossless cho RGB PNG
    elif codec == "ffv1":
        ffmpeg_cmd.extend(["-c:v", "ffv1"])  # Lossless FFV1
        pix_fmt = "bgr0"  # Phù hợp với FFV1
    else:
        raise ValueError(f"Codec không hỗ trợ: {codec}")
    
    ffmpeg_cmd.extend([
        "-r", str(fps),  # Tốc độ khung hình
        "-pix_fmt", pix_fmt,  # Định dạng pixel
        output_video
    ])
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        print(f"Video đã được lưu tại: {output_video} (FFmpeg, {codec} codec)")
        print(f"FFmpeg stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed:\n{e.stderr}")
    finally:
        os.unlink(temp_list.name)
    
    # Kiểm tra kích thước file
    if not os.path.exists(output_video) or os.path.getsize(output_video) == 0:
        raise RuntimeError(f"Video đầu ra {output_video} rỗng hoặc không được tạo!")
    
    video_size = os.path.getsize(output_video) / (1024 * 1024)  # MB
    original_video = "input/test.mp4"
    original_size = os.path.getsize(original_video) / (1024 * 1024) if os.path.exists(original_video) else None
    print(f"Kích thước video đầu ra: {video_size:.2f} MB")
    if original_size:
        print(f"Kích thước video gốc (input/test.mp4): {original_size:.2f} MB")
    
    # Kiểm tra độ dài video
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", output_video
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(json.loads(result.stdout)["format"]["duration"])
        print(f"Độ dài video: {duration:.2f} giây")
    except (subprocess.CalledProcessError, KeyError, ValueError):
        print(f"Cảnh báo: Không thể xác định độ dài video {output_video}")
    
    # Kiểm tra tính toàn vẹn bit cho khung hình giấu tin
    print("\nKiểm tra tính toàn vẹn bit cho khung hình giấu tin...")
    temp_folder = tempfile.mkdtemp()
    try:
        # Tách khung hình từ video
        cap = cv2.VideoCapture(output_video)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {output_video}")
        
        frame_count = 0
        extracted_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count in stego_indices:
                extracted_frame_path = os.path.join(temp_folder, f"frame_{frame_count}.png")
                cv2.imwrite(extracted_frame_path, frame)
                extracted_frames.append((frame_count, extracted_frame_path))
        
        cap.release()
        
        # So sánh khung hình giấu tin
        all_identical = True
        for frame_idx, extracted_frame_path in extracted_frames:
            orig_frame_path = os.path.join(stego_folder, f"frame_{frame_idx}.png")
            orig_frame = cv2.imread(orig_frame_path)
            extracted_frame = cv2.imread(extracted_frame_path)
            if orig_frame is None or extracted_frame is None:
                print(f"Lỗi khi đọc khung hình {frame_idx}: {orig_frame_path} hoặc {extracted_frame_path}")
                all_identical = False
                continue
            if not np.array_equal(orig_frame, extracted_frame):
                print(f"Khung hình {frame_idx} (frame_{frame_idx}.png) không khớp về mặt bit!")
                all_identical = False
            else:
                print(f"Khung hình {frame_idx} (frame_{frame_idx}.png): Bit toàn vẹn")
        
        if all_identical:
            print("TẤT CẢ khung hình giấu tin đều toàn vẹn về mặt bit!")
        else:
            print("CẢNH BÁO: Một số khung hình giấu tin không toàn vẹn! Thử codec ffv1 với .mkv nếu cần.")
    
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Khôi phục video từ khung hình giấu tin và gốc.")
    parser.add_argument("--stego-folder", default="result/stego_frames/", help="Thư mục chứa khung hình giấu tin")
    parser.add_argument("--original-folder", default="input/frames/", help="Thư mục chứa khung hình gốc")
    parser.add_argument("--output-video", default="result/stego_video.mp4", help="Đường dẫn video đầu ra")
    parser.add_argument("--indices-file", default="input/frame_key.txt", help="File chứa các chỉ số khung hình giấu tin")
    parser.add_argument("--fps", type=float, default=None, help="Tốc độ khung hình (mặc định: lấy từ input/test.mp4)")
    parser.add_argument("--codec", default="libx264", choices=["libx264", "ffv1"], help="Codec video (libx264 hoặc ffv1)")
    args = parser.parse_args()
    
    # Lấy fps từ video gốc nếu không chỉ định
    fps = args.fps if args.fps is not None else get_video_fps()
    
    # Điều chỉnh phần mở rộng file dựa trên codec
    output_video = args.output_video
    if args.codec == "ffv1" and output_video.endswith(".mp4"):
        output_video = output_video.replace(".mp4", ".mkv")
    traditional_output = output_video.replace("stego_video", "stego_video_traditional")
    
    try:
        # Khôi phục video cho phương pháp cải tiến
        print("Khôi phục video cho phương pháp cải tiến...")
        reconstruct_video_ffmpeg(
            stego_folder=args.stego_folder,
            original_folder=args.original_folder,
            output_video=output_video,
            indices_file=args.indices_file,
            fps=fps,
            codec=args.codec
        )
        
        # Khôi phục video cho phương pháp truyền thống
        print("\nKhôi phục video cho phương pháp truyền thống...")
        reconstruct_video_ffmpeg(
            stego_folder="result/traditional_stego_frames/",
            original_folder=args.original_folder,
            output_video=traditional_output,
            indices_file=args.indices_file,
            fps=fps,
            codec=args.codec
        )
    except FileNotFoundError:
        print("FFmpeg hoặc ffprobe không được cài đặt hoặc không có trong PATH. Vui lòng cài đặt FFmpeg.")
    except RuntimeError as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()