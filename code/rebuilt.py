import os
import cv2

def read_frame_indices(frame_key_path):
    with open(frame_key_path, 'r') as f:
        indices = [int(x.strip()) for x in f.read().strip().split(',') if x.strip()]
    return set(indices)

def reconstruct_video(
    original_frames_dir="input/frames/",
    stego_frames_dir="result/stego_frames/",
    frame_key_path="input/frame_key.txt",
    output_video_path="result/rebuilt_stego_video.mp4",
    fps=30
):
    # Get all frame filenames sorted by frame number
    frame_files = sorted(
        [f for f in os.listdir(original_frames_dir) if f.startswith("frame_") and f.endswith(".png")],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    # Indices of frames that are stego
    stego_indices = read_frame_indices(frame_key_path)

    # Read first frame to get size
    first_frame_path = os.path.join(original_frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, layers = first_frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for fname in frame_files:
        idx = int(fname.split('_')[1].split('.')[0])
        if idx in stego_indices:
            stego_path = os.path.join(stego_frames_dir, fname)
            if os.path.exists(stego_path):
                frame = cv2.imread(stego_path)
            else:
                print(f"Warning: Stego frame {stego_path} not found, using original.")
                frame = cv2.imread(os.path.join(original_frames_dir, fname))
        else:
            frame = cv2.imread(os.path.join(original_frames_dir, fname))
        out.write(frame)
    out.release()
    print(f"Rebuilt video saved to {output_video_path}")

if __name__ == "__main__":
    reconstruct_video()