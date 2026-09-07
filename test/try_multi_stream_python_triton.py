"""
SCRIPT KIỂM THỬ BƯỚC 2: MULTI-CAMERA STREAM LOADER + PYTHON MINI-TRITON INFERENCE SERVER
--------------------------------------------------------------------------------
Mục đích:
1. Mô phỏng 100% Kiến trúc Triton Inference Server mà KHÔNG CẦN DOCKER hay CẦN QUYỀN ADMIN.
2. 3 Camera Processes đọc video song song (dùng FileVideoStream Producer Thread).
3. 1 Central PythonMiniTritonServer Process chạy 1 Model duy nhất, gom Dynamic Batch 5ms.
4. Đo hiệu năng và kiểm tra tổng thông lượng hệ thống (FPS).

Tuân theo chuẩn thiết kế trong:
- OFFLINE_MULTICAM_TRACKING_GUIDE.md (Công đoạn 2)
- src/main.py (Central Inference Orchestrator)
"""

import os
import sys
import time
from multiprocessing import Process, Queue, set_start_method

from modules.loader.try_file_stream import FileVideoStream
from modules.detection.python_mini_triton import PythonMiniTritonServer, PythonMiniTritonClient

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def run_camera_worker(cam_info: dict, request_queue: Queue, response_queue: Queue, result_queue: Queue):
    """
    Worker Process per Camera:
    1. Đọc stream bằng FileVideoStream (Producer Thread).
    2. Gửi frame sang PythonMiniTritonServer qua Queue và nhận về BBox người.
    """
    cam_id = cam_info["id"]
    video_path = cam_info["path"]
    cam_fps = cam_info.get("cam_fps", 30)
    start_time_str = cam_info.get("start_time", "2026-08-18 10:00:00")

    print(f"[{cam_id}] ===> TIẾN TRÌNH CAMERA BẮT ĐẦU (PID: {os.getpid()})")

    # 1. Khởi tạo Stream Loader trong Tiến trình con của Camera
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

    # 2. Khởi tạo Python Mini-Triton Client
    client = PythonMiniTritonClient(cam_id, request_queue, response_queue)

    worker_start_time = time.time()
    read_count = 0
    total_detected_boxes = 0

    # 3. Vòng lặp Consumer AI Inference
    while fvs.is_running():
        if not fvs.has_frame():
            time.sleep(0.005)
            continue

        timestamp, frame_idx, frame = fvs.read()
        read_count += 1

        # GỬI FRAME SANG CENTRAL PYTHON MINI-TRITON SERVER VÀ NHẬN KẾT QUẢ
        detections = client.detect(frame_idx, frame)
        total_detected_boxes += len(detections)

        # Log định kỳ tiến trình
        if read_count % 30 == 0 or read_count == 1:
            time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
            ms = int((timestamp % 1) * 1000)
            box_info = f"Found {len(detections)} 'person' BBox(es)" if detections else "No BBox"
            print(f" -> [{cam_id}] Frame #{frame_idx:03d} | Timestamp: {time_str}.{ms:03d} | AI Detect: {box_info}")

    worker_elapsed = time.time() - worker_start_time
    fps = read_count / (worker_elapsed + 1e-6)

    # 4. Dừng loader
    fvs.stop()
    print(f"[{cam_id}] <=== HOÀN THÀNH: {read_count} frames | {total_detected_boxes} BBoxes | {fps:.1f} FPS")

    # 5. Gửi kết quả báo cáo về Master
    result_queue.put({
        "cam_id": cam_id,
        "frames": read_count,
        "boxes": total_detected_boxes,
        "elapsed": worker_elapsed,
        "fps": fps
    })


def main():
    if sys.platform == 'win32':
        set_start_method('spawn', force=True)

    print("================================================================================")
    print("      BENCHMARK MULTI-CAMERA + PYTHON MINI-TRITON INFERENCE SERVER")
    print("================================================================================")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cam1_path = os.path.join(base_dir, "synthetic_cam1.mp4")
    cam2_path = os.path.join(base_dir, "synthetic_cam2.mp4")
    cam3_path = os.path.join(base_dir, "synthetic_cam3.mp4")

    camera_configs = [
        {"id": "CAM_01", "path": cam1_path, "cam_fps": 30, "start_time": "2026-08-18 10:00:00"},
        {"id": "CAM_02", "path": cam2_path, "cam_fps": 30, "start_time": "2026-08-18 10:00:00"},
        {"id": "CAM_03", "path": cam3_path, "cam_fps": 30, "start_time": "2026-08-18 10:00:00"},
    ]

    # 1. Khởi tạo các Hàng đợi giao tiếp (Inter-Process Communication Queues)
    request_queue = Queue()
    response_queues = {cam["id"]: Queue() for cam in camera_configs}
    result_queue = Queue()

    print("[MASTER] Đang khởi tạo Tiến trình AI Trung tâm (Python Mini-Triton Server)...")
    master_start_time = time.time()

    # 2. Khởi tạo và Bật Tiến trình AI Trung tâm (Tương đương Triton Server)
    ai_server_process = PythonMiniTritonServer(request_queue, response_queues, max_batch_size=8, max_delay_sec=0.005)
    ai_server_process.start()

    # 3. Khởi tạo và Bật các Tiến trình Camera Workers
    workers = []
    for cam_info in camera_configs:
        cam_id = cam_info["id"]
        p = Process(
            target=run_camera_worker,
            args=(cam_info, request_queue, response_queues[cam_id], result_queue)
        )
        workers.append(p)

    for p in workers:
        p.start()

    # Đợi các camera hoàn thành
    for p in workers:
        p.join()

    total_master_time = time.time() - master_start_time

    # Dừng các camera worker processes
    for p in workers:
        if p.is_alive():
            p.terminate()
            p.join(timeout=0.5)

    request_queue.put("STOP")
    if ai_server_process.is_alive():
        ai_server_process.join(timeout=0.5)
        if ai_server_process.is_alive():
            ai_server_process.terminate()

    # Tổng hợp kết quả

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    results.sort(key=lambda x: x["cam_id"])

    print("\n================================================================================")
    print("      KẾT QUẢ KIỂM THỬ: MULTI-CAM + PYTHON MINI-TRITON INFERENCE SERVER")
    print("================================================================================")
    for r in results:
        print(f"  * {r['cam_id']}: {r['frames']} frames | {r['boxes']} BBoxes detected | {r['fps']:.1f} FPS")

    total_frames = sum(r["frames"] for r in results)
    total_boxes = sum(r["boxes"] for r in results)
    system_fps = total_frames / (total_master_time + 1e-6)

    print("--------------------------------------------------------------------------------")
    print(f"Tổng số Camera                : {len(camera_configs)} Cams")
    print(f"Tổng số Frame đã xử lý         : {total_frames} frames")
    print(f"Tổng số Person BBox phát hiện   : {total_boxes} boxes")
    print(f"Thời gian chạy (Wall Clock)    : {total_master_time:.2f} giây")
    print(f"THROUGHPUT TOÀN HỆ THỐNG        : {system_fps:.1f} TOTAL FPS")
    print("================================================================================")
    print("Thành công! Hệ thống chạy chuẩn kiến trúc Triton mà không cần Docker hay quyền root!")


if __name__ == '__main__':
    main()
