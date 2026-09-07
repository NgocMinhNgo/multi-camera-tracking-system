"""
Script này thử nghiệm luồng đọc video bằng FileVideoStream.
"""
import sys
import os
import cv2
import time
import numpy as np

from modules.loader.try_file_stream import FileVideoStream

# Thiết lập UTF-8 encoding cho Windows Terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def run_test():
    test_video_path = r"C:\Users\Admin\Downloads\tmp_prj\synthetic_test_video.mp4"


    # Khởi tạo FileVideoStream đọc file video từ đĩa bằng background thread
    fvs = FileVideoStream(
        path=test_video_path,
        cam_fps=30,
        fps=30,
        queue_size=20,
        current_time="2026-08-18 10:00:00", # cái này cần xem xét nhé
        backend="FFmpeg"
    )

    # Bắt đầu luồng Producer Thread, tk producer đang chạy
    # chú thỏ chạy trước
    fvs.start()
    time.sleep(0.2)

    # bắt đầu đọc (consumer bắt đầu đọc, trong vòng while)
    start_time = time.time()
    read_count = 0 # biến cục bộ, thuộc hàm này, không thuộc bất cứ class nào

    while fvs.is_running(): # consumer chạy sau 1 chút - rùa bắt đầu chạy song song với thỏ, đọc các frame mà thỏ đặt vào trong queue
        # cái .is_running ý là muốn check đồng thời 2 điều kiện là rổ còn carot hay không hoặc thỏ còn đang chạy hay không

        if not fvs.has_frame(): # tức quêu rỗng
            time.sleep(0.01)
            continue
            
        # Lấy tuple (timestamp, frame_index, frame_matrix) từ Hàng đợi
        timestamp, frame_idx, frame = fvs.read() # chính là Q.get() # trở thành biến cục bộ trong hàm
        read_count += 1

        # giả lập thời gian AI xử lý (giả sử ~100ms/frame)
        time.sleep(0.1)

        # in ra frame đã đọc được để mà còn check tiến trình
        if read_count % fvs.keep_frame_interval == 0 or read_count ==1:
            time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
            ms = int((timestamp % 1) * 1000)
            print(f" -> Consumer da boc Frame #{frame_idx} | Timestamp: {time_str}.{ms:03d} | Size: {frame.shape}")

    total_time = time.time() - start_time
    fps_achieved = read_count / (total_time + 1e-6)

    # 5. Dừng luồng đọc video
    fvs.stop()

    print(f"Ket qua kiem thu thanh cong!")
    print(f"- Tong so frame da doc tu Queue: {read_count} frames")
    print(f"- Thoi gian xu ly: {total_time:.2f} giay")
    print(f"- Toc do doc (Throughput): {fps_achieved:.1f} FPS")

if __name__ == '__main__':
    run_test()




