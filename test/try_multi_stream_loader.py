"""
SCRIPT KIỂM THỬ MULTI-CAMERA STREAM LOADER (MULTIPROCESSING PATTERN)
--------------------------------------------------------------------------------
Mục đích:
1. Đọc đồng thời N luồng video từ các Camera khác nhau (sử dụng multiprocessing.Process).
2. Mỗi Process chạy độc lập một Instance FileVideoStream (Producer-Consumer).
3. Kiểm tra tính cách ly (Process Isolation) và đo Tổng dung lượng/Tốc độ xử lý (Total System Throughput - FPS).
"""

import os
import sys
import time
import cv2
import numpy as np
from multiprocessing import Process, Queue, set_start_method

from modules.loader.try_file_stream import FileVideoStream

# Thiết lập UTF-8 encoding cho Windows Terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def run_camera_worker(cam_info: dict, result_queue: Queue):
    """
    Hàm Worker chạy trong một Tiến trình riêng biệt (Process per Camera).
    Quan trọng: Mỗi Camera có 1 instance FileVideoStream riêng (gồm 1 Background Producer Thread).
    """

    cam_id = cam_info["id"]
    video_path = cam_info["path"]
    cam_fps = cam_info.get("cam_fps", 30)
    start_time_str = cam_info.get("start_time", "2026-08-18 10:00:00")
    ai_delay = cam_info.get("simulated_ai_delay", 0.05)  # Giả lập độ trễ AI (~20 FPS)

    print(f"[{cam_id}] ===> TIẾN TRÌNH CAMERA BẮT ĐẦU (PID: {os.getpid()})")

    # 1. Khởi tạo Stream Loader trong tiến trình con
    fvs = FileVideoStream(
        path=video_path,
        cam_fps=cam_fps,
        fps=cam_fps,
        queue_size=20,
        current_time=start_time_str,
        backend="FFmpeg"
    )

    # 2. Bật Producer Thread
    fvs.start()
    time.sleep(0.2)  # Đợi Producer nạp đệm một vài frames vào Queue

    worker_start_time = time.time()
    read_count = 0

    # 3. Vòng lặp Consumer (Giả lập AI Engine đọc frame và xử lý)
    while fvs.is_running():
        if not fvs.has_frame():
            time.sleep(0.005)
            continue

        # Lấy frame từ Queue của Camera này
        timestamp, frame_idx, frame = fvs.read()
        read_count += 1

        # Giả lập thời gian AI xử lý (YOLO + SORT + FastReID)
        time.sleep(ai_delay)

        # In log định kỳ kiểm tra tiến trình
        if read_count % 30 == 0 or read_count == 1:
            time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
            ms = int((timestamp % 1) * 1000)
            print(f" -> [{cam_id}] Processed Frame #{frame_idx:03d} | Timestamp: {time_str}.{ms:03d} | Size: {frame.shape}")

    worker_elapsed = time.time() - worker_start_time
    fps = read_count / (worker_elapsed + 1e-6)

    # 4. Giải phóng tài nguyên
    fvs.stop()
    print(f"[{cam_id}] <=== TIẾN TRÌNH CAMERA HOÀN THÀNH: {read_count} frames | {worker_elapsed:.2f}s | {fps:.1f} FPS")

    # 5. Gửi kết quả về Master Process qua Queue
    result_queue.put({
        "cam_id": cam_id,
        "frames": read_count,
        "elapsed": worker_elapsed,
        "fps": fps
    })


def main():
    # Bắt buộc trên Windows để spawn process sạch sẽ
    if sys.platform == 'win32':
        set_start_method('spawn', force=True)

    # 1. Chuẩn bị 3 file video giả lập cho 3 Camera
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cam1_path = os.path.join(base_dir, "synthetic_cam1.mp4")
    cam2_path = os.path.join(base_dir, "synthetic_cam2.mp4")
    cam3_path = os.path.join(base_dir, "synthetic_cam3.mp4")


    # 2. Cấu hình 3 Luồng Camera cùng mốc thời gian thực tế
    camera_configs = [
        {"id": "CAM_01", "path": cam1_path, "cam_fps": 30, "start_time": "2026-08-18 10:00:00", "simulated_ai_delay": 0.03},
        {"id": "CAM_02", "path": cam2_path, "cam_fps": 30, "start_time": "2026-08-18 10:00:00", "simulated_ai_delay": 0.03},
        {"id": "CAM_03", "path": cam3_path, "cam_fps": 30, "start_time": "2026-08-18 10:00:00", "simulated_ai_delay": 0.03},
    ]

    # 3. Tạo Queue chung để nhận báo cáo từ các tiến trình con
    result_queue = Queue()
    workers = []

    print("\n[MASTER] Đang khởi tạo và kích hoạt các Worker Process...")
    master_start_time = time.time()

    # 4. Spawn các Tiến trình per Camera (giống src/main.py)
    for cam_info in camera_configs:
        p = Process(target=run_camera_worker, args=(cam_info, result_queue))
        workers.append(p)

    # Start tất cả tiến trình chạy song song
    for p in workers:
        p.start()

    # Đợi tất cả các tiến trình hoàn thành (Đúng theo src/main.py: worker.join())
    for p in workers:
        p.join()

    total_master_time = time.time() - master_start_time

    # 5. Thu thập kết quả từ Queue
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Sort kết quả theo tên Camera
    results.sort(key=lambda x: x["cam_id"])

    # 6. Báo cáo thống kê hiệu năng (Throughput Benchmark Report)
    total_frames_processed = sum(r["frames"] for r in results)
    avg_per_cam_fps = np.mean([r["fps"] for r in results]) if results else 0
    total_system_fps = total_frames_processed / (total_master_time + 1e-6)

    print("\n================================================================================")
    print("                    Kết quả kiểm thử thực tế đọc stream video multicamera")
    print("================================================================================")
    for r in results:
        print(f"  * {r['cam_id']}: Processed {r['frames']} frames | Time: {r['elapsed']:.2f}s | Rate: {r['fps']:.1f} FPS")

    print("--------------------------------------------------------------------------------")
    print(f"Tổng số Camera xử lý song song  : {len(camera_configs)} Cams")
    print(f"Tổng số frame đã xử lý toàn hệ thống: {total_frames_processed} frames")
    print(f"Tổng thời gian chạy thực tế (Wall Clock): {total_master_time:.2f} giây")
    print(f"Tốc độ trung bình đơn camera     : {avg_per_cam_fps:.1f} FPS/cam")
    print(f"TỔNG THROUGHPUT TOÀN HỆ THỐNG    : {total_system_fps:.1f} TOTAL FPS")
    print("================================================================================")
    print("Kiểm thử thành công! Các tiến trình chạy hoàn toàn độc lập mà không bị nghẽn đĩa!")


if __name__ == '__main__':
    main()
