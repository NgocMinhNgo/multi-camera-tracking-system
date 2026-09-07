"""
SCRIPT KIỂM THỬ TRỰC QUAN REALTIME: MULTI-CAMERA STREAM + PYTHON MINI-TRITON + DRAW BBOX
--------------------------------------------------------------------------------
Mục đích:
1. Đọc 3 luồng video song song (Multiprocessing Camera Workers + Producer Threads).
2. Gọi PythonMiniTritonServer để phát hiện ô Bounding Box người.
3. Vẽ ô Bounding Box màu rực rỡ + nhãn ID/Score đè lên khung hình của từng Camera.
4. Ghép 3 luồng camera nằm ngang (Horizontal Grid) và hiển thị REALTIME lên màn hình (cv2.imshow).

Tuân theo chuẩn thiết kế trong:
- OFFLINE_MULTICAM_TRACKING_GUIDE.md (Công đoạn 5 - Visualization)
"""

import os
import sys
import time
import cv2
import numpy as np
from multiprocessing import Process, Queue, set_start_method

from modules.loader.try_file_stream import FileVideoStream
from modules.detection.python_mini_triton import PythonMiniTritonServer, PythonMiniTritonClient

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def draw_bounding_boxes(frame: np.ndarray, detections: list, cam_id: str):
    """Vẽ Bounding Box + Label màu sắc rực rỡ lên khung hình."""
    annotated_frame = frame.copy()

    # Màu sắc riêng cho từng camera (BGR format)
    cam_colors = {
        "CAM_0351": (0, 255, 0),     # Xanh lá bright
        "CAM_0353": (255, 255, 0),   # Cyan
        "CAM_0358": (255, 0, 255)    # Magenta
    }
    box_color = cam_colors.get(cam_id, (0, 255, 0))

    for det in detections:
        x1, y1, x2, y2 = det.box
        score = det.score
        classname = det.classname

        # 1. Vẽ ô Bounding Box hình chữ nhật
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

        # 2. Vẽ nhãn Text mờ nền (Background Header Box)
        label = f"{cam_id} | {classname} {score:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Vẽ hình chữ nhật nền cho chữ
        cv2.rectangle(annotated_frame, (x1, y1 - h - 12), (x1 + w + 10, y1), box_color, -1)
        # In chữ màu đen nổi bật trên nền
        cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # In thông tin góc trên bên trái của Camera
    header_text = f"LIVE STREAM: {cam_id} ({len(detections)} people)"
    cv2.putText(annotated_frame, header_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    return annotated_frame


def run_camera_worker_vis(cam_info: dict, request_queue: Queue, response_queue: Queue, vis_queue: Queue):
    """
    Worker Process cho từng Camera:
    1. Đọc frame từ FileVideoStream.
    2. Gửi sang AI Server lấy BBox.
    3. Vẽ BBox đè lên frame.
    4. Đẩy frame đã vẽ vào vis_queue cho Master hiển thị màn hình Realtime.
    """
    cam_id = cam_info["id"]
    video_path = cam_info["path"]
    cam_fps = cam_info.get("cam_fps", 30)
    start_time_str = cam_info.get("start_time", "2026-08-18 10:00:00")

    print(f"[{cam_id}] ===> Bắt đầu Tiến trình Camera + Rendering (PID: {os.getpid()})")

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

    while fvs.is_running():
        if not fvs.has_frame():
            time.sleep(0.005)
            continue

        timestamp, frame_idx, frame = fvs.read()

        # 1. Gọi AI Server lấy BBox
        detections = client.detect(frame_idx, frame)

        # 2. Vẽ BBox đè lên frame
        annotated_frame = draw_bounding_boxes(frame, detections, cam_id)

        # 3. Đẩy frame đã vẽ sang Queue hiển thị
        vis_queue.put((cam_id, frame_idx, annotated_frame))

    fvs.stop()
    print(f"[{cam_id}] <=== Tiến trình hoàn thành!")
    vis_queue.put((cam_id, -1, None))  # Tín hiệu kết thúc cho camera này


def main():
    if sys.platform == 'win32':
        set_start_method('spawn', force=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cam351_path = os.path.join(base_dir, "short_data", "camera_0351", "video.mp4")
    cam353_path = os.path.join(base_dir, "short_data", "camera_0353", "video.mp4")
    cam358_path = os.path.join(base_dir, "short_data", "camera_0358", "video.mp4")

    camera_configs = [
        {"id": "CAM_0351", "path": cam351_path, "cam_fps": 30, "start_time": "2026-08-18 10:00:00"},
        {"id": "CAM_0353", "path": cam353_path, "cam_fps": 30, "start_time": "2026-08-18 10:00:00"},
        {"id": "CAM_0358", "path": cam358_path, "cam_fps": 30, "start_time": "2026-08-18 10:00:00"},
    ]

    request_queue = Queue()
    response_queues = {cam["id"]: Queue() for cam in camera_configs}
    vis_queue = Queue()

    print("================================================================================")
    print("   REALTIME MULTI-CAMERA BBOX MONITORING (REAL VIDEOS: CAM 351, 353, 358)")
    print("================================================================================")
    print("  * Màn hình hiển thị 3 luồng video thực tế ghép song song (16:9).")
    print("  * Mỗi camera hiển thị Bounding Box phát hiện người theo thời gian thực.")
    print("  * Nhấn phím 'q' hoặc 'ESC' trên cửa sổ OpenCV để thoát màn hình.")
    print("================================================================================")

    # 1. Bật AI Server Process
    ai_server_process = PythonMiniTritonServer(request_queue, response_queues, max_batch_size=8, max_delay_sec=0.005)
    ai_server_process.start()

    # 2. Bật 3 Camera Worker Processes
    workers = []
    for cam_info in camera_configs:
        p = Process(
            target=run_camera_worker_vis,
            args=(cam_info, request_queue, response_queues[cam_info["id"]], vis_queue)
        )
        workers.append(p)

    for p in workers:
        p.start()

    # 3. VÒNG LẶP MASTER HIỂN THỊ REALTIME TRÊN MÀN HÌNH (CV2.IMSHOW)
    latest_frames = {"CAM_0351": None, "CAM_0353": None, "CAM_0358": None}
    active_cams = {"CAM_0351": True, "CAM_0353": True, "CAM_0358": True}

    window_name = "MULTI-CAMERA REALTIME BBOX MONITORING - REAL SCENE 040"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1440, 270)

    target_size = (480, 270)  # Tỷ lệ 16:9 chuẩn cho video 1080p thực tế

    while any(active_cams.values()):
        # Lấy các frame mới nhất từ vis_queue
        while not vis_queue.empty():
            cam_id, frame_idx, frame = vis_queue.get()
            if frame_idx == -1:
                active_cams[cam_id] = False
            else:
                latest_frames[cam_id] = frame

        # Ghép 3 khung hình camera nằm ngang
        cam_views = []
        for cam_id in ["CAM_0351", "CAM_0353", "CAM_0358"]:
            if latest_frames[cam_id] is not None:
                resized = cv2.resize(latest_frames[cam_id], target_size)
            else:
                # Nếu chưa có frame -> Nền đen có chữ Offline
                resized = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                cv2.putText(resized, f"{cam_id} LOADING...", (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cam_views.append(resized)

        # Ghép ngang 3 bức ảnh (Horizontal Stitching)
        grid_canvas = np.hstack(cam_views)

        # Vẽ dải phân cách trắng giữa các camera
        cv2.line(grid_canvas, (target_size[0], 0), (target_size[0], target_size[1]), (255, 255, 255), 2)
        cv2.line(grid_canvas, (target_size[0] * 2, 0), (target_size[0] * 2, target_size[1]), (255, 255, 255), 2)

        # Hiển thị lên màn hình desktop
        cv2.imshow(window_name, grid_canvas)

        # Đợi 30ms (~33 FPS) và kiểm tra phím bấm thoát ('q' hoặc ESC)
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            print("\n[USER] Đã nhấn 'q'/ESC. Dừng hiển thị Realtime...")
            break


    # Dọn dẹp cửa sổ OpenCV và kết thúc các tiến trình sạch sẽ
    cv2.destroyAllWindows()

    # Dừng các Camera Worker Processes ngay lập tức (tránh bị kẹt chờ response queue)
    for p in workers:
        if p.is_alive():
            p.terminate()
            p.join(timeout=0.5)

    request_queue.put("STOP")
    if ai_server_process.is_alive():
        ai_server_process.join(timeout=0.5)
        if ai_server_process.is_alive():
            ai_server_process.terminate()

    print("[SUCCESS] Đã dừng toàn bộ hệ thống mượt mà và trả lại Terminal!")


if __name__ == '__main__':
    main()

