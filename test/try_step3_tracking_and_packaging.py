# -*- coding: utf-8 -*-
"""
SCRIPT KIỂM THỬ BƯỚC 3: SINGLE-CAM TRACKING, RAM BUFFER & TRACKLET PACKAGING
--------------------------------------------------------------------------------
Mục đích:
1. Đọc 3 luồng video thực tế song song (CAM 351, CAM 353, CAM 358).
2. Gọi PythonMiniTriton AI Server chạy YOLOv8 trên GPU 5 (cuda:5).
3. Triển khai SORT Tracker (Kalman Filter + Hungarian IoU) gán ID tạm thời duy nhất cho từng camera.
4. Quản lý RAM đệm (PersonTrackManager) tích lũy ảnh crop nét (is_good_box) của từng đối tượng.
5. Khi người bước ra khỏi camera (mất dấu > max_fragment_time):
   - Kích hoạt FastReID Vectorizer (trích vector đặc trưng 256 chiều).
   - Kích hoạt Tracklet Clusterer (gọt lọc Top 5 vector đại diện).
   - Đóng gói dữ liệu TrackletPackage và lưu ra đĩa (JSON payload).
   - Giải phóng RAM memory (remove_track).
6. Hiển thị trực quan và lưu:
   - File video ghép 3 camera: output_step3_tracking_vis.mp4
   - 3 File video riêng biệt cho từng camera: output_step3_CAM_0351.mp4, output_step3_CAM_0353.mp4, output_step3_CAM_0358.mp4

Tuân theo chuẩn thiết kế trong:
- OFFLINE_MULTICAM_TRACKING_GUIDE.md (Công đoạn 3)
- README_STEP3_TRACKING_AND_TRACKLET_PACKAGING.md
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from multiprocessing import Process, Queue, Event, set_start_method

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.loader.try_file_stream import FileVideoStream
from modules.detection.python_mini_triton import PythonMiniTritonServer, PythonMiniTritonClient
from modules.tracking import PersonTracker, PersonTrackManager, TrackletPackager

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def get_color_for_id(track_id: int) -> tuple[int, int, int]:
    """Generates a distinct, vibrant BGR color for a given track ID."""
    np.random.seed(int(track_id) * 31 + 7)
    color = np.random.randint(40, 255, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def draw_tracking_boxes(frame: np.ndarray, alive_tracks: list, cam_id: str) -> np.ndarray:
    """Vẽ Bounding Box + Persistent Track ID với màu sắc riêng cho từng đối tượng."""
    annotated_frame = frame.copy()

    for trk in alive_tracks:
        x1, y1, x2, y2 = trk.box
        score = trk.score
        track_id = trk.track_id

        # Unique color per track_id
        box_color = get_color_for_id(track_id)

        # 1. Bounding Box rectangle
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

        # 2. Text label
        label = f"{cam_id} | ID #{track_id} ({score:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

        # Label background
        cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w + 8, y1), box_color, -1)
        # Text label on top
        cv2.putText(annotated_frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Stream header text
    header_text = f"TRACKING: {cam_id} ({len(alive_tracks)} active)"
    cv2.putText(annotated_frame, header_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

    return annotated_frame


def run_camera_worker_step3(cam_info: dict, request_queue: Queue, response_queue: Queue, vis_queue: Queue, output_dir: str, max_frames: int = 300):
    """
    Worker Process cho từng Camera:
    1. Đọc frame từ FileVideoStream.
    2. Gọi AI Server lấy BBox từ YOLOv8.
    3. Chạy PersonTracker (SORT) gán ID tạm.
    4. Cập nhật RAM dict_tracks và lọc ảnh crop nét.
    5. Kiểm tra ID đã chết hẳn -> Trích vector 256-dim -> Đóng gói Tracklet JSON & Giải phóng RAM!
    6. Vẽ BBox đè lên frame và gửi sang vis_queue rendering.
    """
    cam_id = cam_info["id"]
    video_path = cam_info["path"]
    cam_fps = cam_info.get("cam_fps", 30)
    start_time_str = cam_info.get("start_time", "2026-08-18 10:00:00")

    print(f"[{cam_id}] ===> Bắt đầu Tiến trình Camera + Tracking + Packaging (PID: {os.getpid()})")

    fvs = FileVideoStream(
        path=video_path,
        cam_fps=cam_fps,
        fps=cam_fps,
        queue_size=20,
        current_time=start_time_str,
        backend="FFmpeg"
    )
    fvs.start()
    time.sleep(0.2)

    client = PythonMiniTritonClient(cam_id, request_queue, response_queue)

    # Instantiates Step 3 Tracking & Packaging engines
    tracker = PersonTracker(max_age=15, min_hits=2, low_iou_threshold=0.25)
    track_manager = PersonTrackManager(cam_id=cam_id, max_fragment_time=1.0)
    packager = TrackletPackager(output_dir=output_dir)

    step_timestamp = 0.0
    processed_count = 0

    while fvs.is_running() and processed_count < max_frames:
        if not fvs.has_frame():
            time.sleep(0.005)
            continue

        timestamp_str, frame_idx, frame = fvs.read()
        processed_count += 1
        step_timestamp += (1.0 / cam_fps)

        # 1. AI Detector
        detections = client.detect(frame_idx, frame)

        # 2. Single-Camera SORT Tracker
        alive_tracks, dead_tracks = tracker.track(detections, frame)

        # 3. RAM Track Manager update
        track_manager.update_session_tracks(step_timestamp, alive_tracks, dead_tracks, frame)

        # 4. Check for confirmed dead tracks (> max_fragment_time)
        real_dead_tracks = track_manager.get_last_dead_tracks(step_timestamp)
        for dead_info in real_dead_tracks:
            # Package dead tracklet with FastReID vectorization & clustering
            pkg = packager.package_and_clean(dead_info, track_manager)
            packager.save_package(pkg)
            print(f"[{cam_id} | STEP 3] Packaged Dead Track #{pkg.track_id}: {pkg.num_frames} frames, {len(pkg.rep_vectors)} vectors (RAM Cleared)")

        # 5. Render BBox visualization with persistent IDs
        annotated_frame = draw_tracking_boxes(frame, alive_tracks, cam_id)

        # 6. Push to visualization queue
        vis_queue.put((cam_id, frame_idx, annotated_frame))

    fvs.stop()
    print(f"[{cam_id}] <=== Đã xử lý đủ {processed_count} frames! Đang xả nốt các Track còn lại...")

    # Final cleanup for active tracks when video ends
    remaining_ids = list(track_manager.dict_tracks.keys())
    for t_id in remaining_ids:
        dead_info = track_manager.dict_tracks[t_id]
        pkg = packager.package_and_clean(dead_info, track_manager)
        packager.save_package(pkg)
        print(f"[{cam_id} | STEP 3 END] Final Package Track #{pkg.track_id}: {pkg.num_frames} frames, {len(pkg.rep_vectors)} vectors (RAM Cleared)")

    print(f"[{cam_id}] <=== Tiến trình hoàn tất 100%!")
    vis_queue.put((cam_id, -1, None))


def main():
    if sys.platform == 'win32':
        set_start_method('spawn', force=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    cam351_path = os.path.join(project_root, "short_data", "camera_0351", "video.mp4")
    cam353_path = os.path.join(project_root, "short_data", "camera_0353", "video.mp4")
    cam358_path = os.path.join(project_root, "short_data", "camera_0358", "video.mp4")

    camera_configs = [
        {"id": "CAM_0351", "path": cam351_path, "cam_fps": 30, "start_time": "2026-08-18 10:00:00"},
        {"id": "CAM_0353", "path": cam353_path, "cam_fps": 30, "start_time": "2026-08-18 10:00:00"},
        {"id": "CAM_0358", "path": cam358_path, "cam_fps": 30, "start_time": "2026-08-18 10:00:00"},
    ]

    output_dir = os.path.join(project_root, "output_tracklets")

    request_queue = Queue()
    response_queues = {cam["id"]: Queue() for cam in camera_configs}
    vis_queue = Queue()

    print("================================================================================")
    print("   STEP 3: SINGLE-CAM TRACKING, RAM BUFFER & TRACKLET PACKAGING TEST")
    print("================================================================================")
    print("  * Luồng: Loader -> YOLOv8 -> SORT Tracker -> RAM dict_tracks -> FastReID Packager")
    print("  * GPU Target: GPU 5 (cuda:5)")
    print("  * Output Tracklets Directory:", output_dir)
    print("================================================================================")

    ready_event = Event()

    # 1. Start AI Server Process
    ai_server_process = PythonMiniTritonServer(
        request_queue, response_queues, max_batch_size=8, max_delay_sec=0.005, ready_event=ready_event, device="cuda:5"
    )
    ai_server_process.start()

    print("[MASTER] Đang chờ AI Server nạp Model (YOLOv8 PyTorch trên GPU 5)...")
    ready_event.wait()
    print("[MASTER] AI Server sẵn sàng! Kích hoạt các Tiến trình Camera Worker...")

    # 2. Start Camera Worker Processes
    workers = []
    max_test_frames = 300
    for cam_info in camera_configs:
        p = Process(
            target=run_camera_worker_step3,
            args=(cam_info, request_queue, response_queues[cam_info["id"]], vis_queue, output_dir, max_test_frames)
        )
        workers.append(p)

    for p in workers:
        p.start()

    # 3. Master Visualization & Video Writers (Individual Cams + Stitched)
    latest_frames = {"CAM_0351": None, "CAM_0353": None, "CAM_0358": None}
    active_cams = {"CAM_0351": True, "CAM_0353": True, "CAM_0358": True}

    target_size = (480, 270)
    grid_size = (target_size[0] * 3, target_size[1])

    output_video_path = os.path.join(project_root, "output_step3_tracking_vis.mp4")
    print(f"[MASTER] Ghi luồng video ghép 3 Camera: {output_video_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    grid_video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, grid_size)

    # Individual Camera Video Writers
    individual_writers = {}
    for c_id in ["CAM_0351", "CAM_0353", "CAM_0358"]:
        cam_out_path = os.path.join(project_root, f"output_step3_{c_id}.mp4")
        individual_writers[c_id] = {
            "writer": None,
            "path": cam_out_path
        }
        print(f"[MASTER] Đăng ký xuất video riêng cho {c_id}: {cam_out_path}")

    rendered_count = 0

    while any(active_cams.values()):
        got_new = False
        while not vis_queue.empty():
            cam_id, frame_idx, frame = vis_queue.get()
            if frame_idx == -1:
                active_cams[cam_id] = False
            else:
                latest_frames[cam_id] = frame
                got_new = True

                # Write to individual camera video file at original resolution
                if cam_id in individual_writers and frame is not None:
                    writer_info = individual_writers[cam_id]
                    if writer_info["writer"] is None:
                        h_orig, w_orig = frame.shape[:2]
                        writer_info["writer"] = cv2.VideoWriter(writer_info["path"], fourcc, 30.0, (w_orig, h_orig))
                    writer_info["writer"].write(frame)

        if not got_new:
            time.sleep(0.01)
            continue

        # Horizontal stitch
        cam_views = []
        for cam_id in ["CAM_0351", "CAM_0353", "CAM_0358"]:
            if latest_frames[cam_id] is not None:
                resized = cv2.resize(latest_frames[cam_id], target_size)
            else:
                resized = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                cv2.putText(resized, f"{cam_id} LOADING...", (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cam_views.append(resized)

        grid_canvas = np.hstack(cam_views)
        cv2.line(grid_canvas, (target_size[0], 0), (target_size[0], target_size[1]), (255, 255, 255), 2)
        cv2.line(grid_canvas, (target_size[0] * 2, 0), (target_size[0] * 2, target_size[1]), (255, 255, 255), 2)

        grid_video_writer.write(grid_canvas)
        rendered_count += 1

        if rendered_count % 30 == 0:
            print(f" -> [STEP 3 VIS] Đã ghi {rendered_count}/{max_test_frames} frames vào {output_video_path}")

    # Release Video Writers
    grid_video_writer.release()
    for c_id, w_info in individual_writers.items():
        if w_info["writer"] is not None:
            w_info["writer"].release()
            print(f"[MASTER] Đã lưu xong video riêng: {w_info['path']}")

    for p in workers:
        p.join()

    ai_server_process.terminate()

    print("================================================================================")
    print(f"[STEP 3 SUCCESS] Đã lưu file video ghép: {output_video_path}")
    print(f"[STEP 3 SUCCESS] Đã lưu các video riêng từng camera: output_step3_CAM_0351.mp4, output_step3_CAM_0353.mp4, output_step3_CAM_0358.mp4")
    print(f"[STEP 3 SUCCESS] Đã lưu các gói Tracklet JSON trong: {output_dir}")
    print("================================================================================")


if __name__ == '__main__':
    main()
