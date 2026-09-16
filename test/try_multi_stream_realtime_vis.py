# -*- coding: utf-8 -*-
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
from multiprocessing import Process, Queue, Event, set_start_method

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.loader.try_file_stream import FileVideoStream
from modules.detection.python_mini_triton import PythonMiniTritonServer, PythonMiniTritonClient

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def draw_bounding_boxes(frame: np.ndarray, detections: list, cam_id: str):
    """Vẽ Bounding Box + Label với màu sắc riêng biệt cho từng Camera."""
    annotated_frame = frame.copy()

    # Màu sắc riêng cho từng camera (BGR format)
    cam_colors = {
        "CAM_0351": (0, 255, 0),     # Xanh lá bright
        "CAM_0353": (255, 255, 0),   # Cyan
        "CAM_0358": (255, 0, 255)    # Magenta / Pink
    }
    box_color = cam_colors.get(cam_id, (0, 255, 0))

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det.box
        score = det.score
        classname = det.classname
        track_id = getattr(det, 'track_id', None)

        # 1. Vẽ ô Bounding Box hình chữ nhật
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

        # 2. Tạo nhãn Text hiển thị
        if track_id is not None:
            label = f"{cam_id} | ID:{track_id} {score:.2f}"
        else:
            label = f"{cam_id} | {classname} {score:.2f}"

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

        # Vẽ hình chữ nhật nền cho chữ
        cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w + 8, y1), box_color, -1)
        # In chữ màu đen nổi bật trên nền
        cv2.putText(annotated_frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # In thông tin góc trên bên trái của Camera
    header_text = f"LIVE STREAM: {cam_id} ({len(detections)} people)"
    cv2.putText(annotated_frame, header_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

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
    project_root = os.path.dirname(base_dir)
    cam351_path = os.path.join(project_root, "short_data", "camera_0351", "video.mp4")
    cam353_path = os.path.join(project_root, "short_data", "camera_0353", "video.mp4")
    cam358_path = os.path.join(project_root, "short_data", "camera_0358", "video.mp4")

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

    ready_event = Event()

    # 1. Bật AI Server Process
    ai_server_process = PythonMiniTritonServer(
        request_queue, response_queues, max_batch_size=8, max_delay_sec=0.005, ready_event=ready_event, device="cuda:5"
    )
    ai_server_process.start()

    print("[MASTER] Đang chờ AI Server nạp Model (YOLOv8 PyTorch)...")
    ready_event.wait()
    print("[MASTER] AI Server đã nạp xong Model! Kích hoạt các Tiến trình Camera Workers...")

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

    # 3. VÒNG LẶP MASTER HIỂN THỊ REALTIME HOẶC GHI FILE VIDEO (HEADLESS SAFE)
    latest_frames = {"CAM_0351": None, "CAM_0353": None, "CAM_0358": None}
    active_cams = {"CAM_0351": True, "CAM_0353": True, "CAM_0358": True}

    target_size = (480, 270)  # Tỷ lệ 16:9 chuẩn cho video 1080p thực tế
    grid_size = (target_size[0] * 3, target_size[1])

    has_display = bool(os.environ.get("DISPLAY"))
    video_writer = None
    output_video_path = os.path.join(project_root, "output_realtime_vis.mp4")

    # Khởi tạo thông tin Video Writer riêng cho từng Camera
    cam_writers = {}
    for c_id in ["CAM_0351", "CAM_0353", "CAM_0358"]:
        cam_writers[c_id] = {
            "writer": None,
            "path": os.path.join(project_root, f"output_{c_id}.mp4"),
            "img_path": os.path.join(project_root, f"output_{c_id}.jpg")
        }

    if has_display:
        try:
            window_name = "MULTI-CAMERA REALTIME BBOX MONITORING - REAL SCENE 040"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1440, 270)
        except Exception as e:
            print(f"[WARN] OpenCV GUI Error ({e}). Falling back to Headless Video Writer mode.")
            has_display = False

    if not has_display:
        print(f"\n[HEADLESS MODE] Không tìm thấy $DISPLAY (SSH/Server session).")
        print(f"[HEADLESS MODE] Đang ghi luồng video ghép 3 Camera: {output_video_path}")
        print(f"[HEADLESS MODE] Đang ghi 3 video riêng từng Camera: output_CAM_0351.mp4, output_CAM_0353.mp4, output_CAM_0358.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, grid_size)

    rendered_count = 0
    max_headless_frames = 300  # Đầy đủ 10 giây preview video trong chế độ Headless

    while any(active_cams.values()):
        # Lấy các frame mới nhất từ vis_queue
        while not vis_queue.empty():
            cam_id, frame_idx, frame = vis_queue.get()
            if frame_idx == -1:
                active_cams[cam_id] = False
            else:
                latest_frames[cam_id] = frame
                # Ghi luồng video gốc chất lượng cao riêng cho Camera này
                if cam_id in cam_writers and frame is not None:
                    c_info = cam_writers[cam_id]
                    if c_info["writer"] is None:
                        h_f, w_f = frame.shape[:2]
                        fourcc_c = cv2.VideoWriter_fourcc(*'mp4v')
                        c_info["writer"] = cv2.VideoWriter(c_info["path"], fourcc_c, 30.0, (w_f, h_f))
                    c_info["writer"].write(frame)
                    # Lưu ảnh preview khung hình đơn của Camera này
                    cv2.imwrite(c_info["img_path"], frame)

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

        if has_display:
            # Hiển thị lên màn hình desktop
            cv2.imshow(window_name, grid_canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or key == ord('q'):
                print("\n[USER] Đã nhấn 'q'/ESC. Dừng hiển thị Realtime...")
                break
        else:
            if video_writer is not None:
                # Chỉ ghi và đếm frame khi đã nhận được luồng video từ ít nhất 1 camera
                if any(f is not None for f in latest_frames.values()):
                    video_writer.write(grid_canvas)
                    rendered_count += 1
                    if rendered_count == 30 or rendered_count == 150:
                        output_img_path = os.path.join(project_root, "output_preview.jpg")
                        cv2.imwrite(output_img_path, grid_canvas)
                    if rendered_count % 30 == 0:
                        print(f" -> [HEADLESS RENDER] Đã ghi {rendered_count}/{max_headless_frames} frames vào output_realtime_vis.mp4")
                    if rendered_count >= max_headless_frames:
                        print(f"[HEADLESS RENDER] Đã ghi đủ {max_headless_frames} frames. Dừng tiến trình ghi.")
                        break
            time.sleep(0.01)

    if has_display:
        cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.release()
        output_img_path = os.path.join(project_root, "output_preview.jpg")
        cv2.imwrite(output_img_path, grid_canvas)
        print(f"[HEADLESS SUCCESS] Đã lưu file video ghép 3 camera thành công: {output_video_path}")
        print(f"[HEADLESS SUCCESS] Đã lưu ảnh chụp khung hình preview: {output_img_path}")

    # Đóng tất cả Video Writer riêng từng Camera
    for c_id, c_info in cam_writers.items():
        if c_info["writer"] is not None:
            c_info["writer"].release()
            print(f"[SUCCESS] Đã lưu video riêng full HD cho {c_id}: {c_info['path']}")
            print(f"[SUCCESS] Đã lưu ảnh preview riêng cho {c_id}: {c_info['img_path']}")

    # Tự động convert video sang chuẩn H.264 (Playable trên 100% QuickTime / Windows / Chrome)
    try:
        import subprocess
        print("\n[POST-PROCESSING] Đang chuyển đổi video sang chuẩn H.264 (Universal Playable format)...")
        for c_id in ["CAM_0351", "CAM_0353", "CAM_0358"]:
            src = os.path.join(project_root, f"output_{c_id}.mp4")
            dst = os.path.join(project_root, f"output_{c_id}_play.mp4")
            if os.path.exists(src):
                subprocess.run(["ffmpeg", "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p", dst], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f" -> [H.264 READY] {dst}")

        src_grid = output_video_path
        dst_grid = os.path.join(project_root, "output_realtime_vis_play.mp4")
        if os.path.exists(src_grid):
            subprocess.run(["ffmpeg", "-y", "-i", src_grid, "-vcodec", "libx264", "-pix_fmt", "yuv420p", dst_grid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f" -> [H.264 READY] {dst_grid}")
    except Exception as e:
        print(f"[WARN] Không thể tự động convert ffmpeg ({e}).")

    # Dừng các Camera Worker Processes sạch sẽ
    for p in workers:
        if p.is_alive():
            p.terminate()
            p.join(timeout=0.1)

    request_queue.put("STOP")
    if ai_server_process.is_alive():
        ai_server_process.terminate()
        ai_server_process.join(timeout=0.1)

    print("\n[SUCCESS] Đã xử lý xong và lưu tất cả 4 video chuẩn H.264 / 4 ảnh preview thành công!")
    sys.stdout.flush()


if __name__ == '__main__':
    main()

