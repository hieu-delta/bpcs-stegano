import os
import shutil

def process_files(source_dir, dest_dir):
    # Chuẩn hóa đường dẫn
    source_dir = os.path.abspath(source_dir)
    dest_dir = os.path.abspath(dest_dir)

    # Kiểm tra thư mục nguồn tồn tại
    if not os.path.exists(source_dir):
        return

    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(dest_dir, exist_ok=True)

    # Di chuyển tất cả file
    for file_name in os.listdir(source_dir):
        if not os.path.isfile(os.path.join(source_dir, file_name)):
            continue  # Bỏ qua nếu không phải file

        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)

        # Di chuyển file
        try:
            shutil.move(source_path, dest_path)
        except Exception:
            pass