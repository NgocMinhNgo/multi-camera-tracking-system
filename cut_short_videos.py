# -*- coding: utf-8 -*-
"""
SCRIPT TRÍCH XUẤT VIDEO NGẮN (1 PHÚT) TỪ SCENE REAL DATA
--------------------------------------------------------------------------------
Mục đích:
1. Quét 5 folder camera trong <project_root>/data/scene040.
2. Trích xuất 1 phút đầu tiên (60 giây) của từng video.
3. Lưu video ngắn tương ứng vào <project_root>/short_data/<camera_id>/video.mp4.
"""

import os
import sys
import time
import cv2

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def cut_video(input_path: str, output_path: str, duration_sec: int = 60):
    """Cắt video lấy N giây đầu tiên và lưu vào output_path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở file video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames_to_cut = int(fps * duration_sec)
    frames_to_write = min(max_frames_to_cut, total_frames)

    print(f"\n[PROCESSING] File: {input_path}")
    print(f" -> Cấu hình gốc: {width}x{height} | {fps:.1f} FPS | Tổng: {total_frames} frames ({total_frames/fps/60:.1f} phút)")
    print(f" -> Tiến hành cắt: {duration_sec} giây đầu (~{frames_to_write} frames) ──► {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_time = time.time()
    count = 0

    while cap.isOpened() and count < frames_to_write:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        out.write(frame)
        count += 1

        if count % 300 == 0 or count == frames_to_write:
            percent = (count / frames_to_write) * 100
            print(f"    - Đã ghi: {count}/{frames_to_write} frames ({percent:.1f}%)")

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[SUCCESS] Đã tạo thành công: {output_path} ({file_size_mb:.1f} MB | {elapsed:.2f}s)")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, "data", "scene040")
    target_dir = os.path.join(base_dir, "short_data")

    if not os.path.exists(source_dir):
        print(f"[ERROR] Không tìm thấy thư mục nguồn: {source_dir}")
        return

    # Liệt kê các thư mục camera_xxxx
    cam_folders = sorted([
        f for f in os.listdir(source_dir) 
        if os.path.isdir(os.path.join(source_dir, f)) and f.startswith("camera_")
    ])

    print("================================================================================")
    print("      SCRIPT TRÍCH XUẤT VIDEO NGẮN (1 PHÚT) CHO 5 CAMERA SCENE040")
    print("================================================================================")
    print(f"Thư mục nguồn : {source_dir}")
    print(f"Thư mục đích  : {target_dir}")
    print(f"Tìm thấy      : {len(cam_folders)} camera ({', '.join(cam_folders)})")
    print("================================================================================")

    total_start = time.time()

    for cam_folder in cam_folders:
        cam_src_dir = os.path.join(source_dir, cam_folder)
        
        # Tìm file video trong camera_folder (thường là video.mp4)
        video_files = [f for f in os.listdir(cam_src_dir) if f.endswith(('.mp4', '.mkv', '.avi'))]
        if not video_files:
            print(f"[WARN] Bỏ qua {cam_folder}: Không tìm thấy file video nào.")
            continue

        video_filename = video_files[0]
        input_video_path = os.path.join(cam_src_dir, video_filename)
        output_video_path = os.path.join(target_dir, cam_folder, video_filename)

        # Cắt 60 giây (1 phút)
        cut_video(input_video_path, output_video_path, duration_sec=60)

    print("\n================================================================================")
    print(f"[COMPLETE] Đã trích xuất xong toàn bộ 5 video ngắn trong {time.time() - total_start:.2f} giây!")
    print(f"Vị trí lưu: {target_dir}")
    print("================================================================================")


if __name__ == '__main__':
    import numpy as np
    main()
