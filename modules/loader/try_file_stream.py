# -*- coding: utf-8 -*-
"""
FILE VIDEO STREAM LOADER (THREADED PRODUCER-CONSUMER PATTERN) - con thỏ
--------------------------------------------------------------------------------
Mục đích: Đọc luồng khung hình (video frames) từ file đĩa (.mp4 / .mkv) bằng một
Tiến trình con (Background Thread) và nạp vào Hàng đợi (Queue) bộ nhớ đệm.

Tư duy First Principles:
1. Đọc đĩa I/O (cv2.VideoCapture.read) là tác vụ nghẽn I/O.
2. Nếu vòng lặp chính của AI (Inference) vừa đọc đĩa vừa chạy AI, GPU sẽ bị chờ đĩa.
3. Giải pháp: Tạo Thread Producer chạy đọc đĩa liên tục nạp vào Queue(maxsize=50).
   Vòng lặp AI (Consumer) chỉ việc bốc frame từ Queue ra xử lý với tốc độ tối đa!
"""


import cv2
import time
from threading import Thread
from queue import Queue
from datetime import datetime

class FileVideoStream:
    """
    Class quản lý đọc luồng video từ file đĩa sử dụng Background Thread.
    """
    def __init__(self, path, cam_fps=30, fps=30, queue_size=50, current_time=None, backend='FFmpeg'):
        """
        Khởi tạo FileVideoStream.

        Args:
            path (str): Đường dẫn tới file video (.mp4, .mkv, .avi)
            cam_fps (int): Tốc độ khung hình thực tế của video gốc (mặc định 30 FPS)
            fps (int): Tốc độ khung hình mong muốn trích xuất (nếu fps < cam_fps sẽ bỏ bớt frame)
            queue_size (int): Kích thước tối đa của Hàng đợi bộ nhớ đệm (mặc định 50 frames)
            current_time (str/float): Mốc thời gian bắt đầu thực tế của video (để gắn timestamp cho từng frame)
            backend (str): Thư viện đọc video ('FFmpeg' cho OpenCV thông thường hoặc 'GStreamer')
        """

        self.path = path
        self.backend = backend
        
        #if backend = ffmpeg
        # 1. Khởi tạo kết nối OpenCV VideoCapture
        self.stream = cv2.VideoCapture(path, cv2.CAP_FFMPEG)

        # Kiểm tra xem file video có mở thành công không
        if not self.stream.isOpened():
            raise ValueError(f"[ERROR] Không thể mở file video tại đường dẫn: {path}")

        # 6. Khởi tạo Hàng đợi (Queue) thread-safe lưu trữ các frame đã đọc
        self.Q = Queue(maxsize=queue_size)

        # 7. Khởi tạo Background Thread (Producer Thread)
        self.thread = Thread(target=self._update, args=())
        self.thread.daemon = True
        
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))

        # 3. Quản lý FPS và cờ tạm dừng thread
        self.stopped = False
        self.cam_fps = max(cam_fps, fps)
        self.fps = min(fps, cam_fps)
        self.fps_sleep = 1.0 / self.fps  # Thời gian nghỉ khi Hàng đợi bị đầy (full)
        
        # 4. Tính toán tỷ lệ bỏ bớt frame (Frame Skipping) nếu fps < cam_fps
        if self.cam_fps == self.fps:
            self.keep_frame_interval = -1  # Lấy 100% tất cả các frame
        else:
            self.keep_frame_interval = round(self.cam_fps / self.fps)

        # 5. Xử lý mốc thời gian Timestamp thực tế
        if current_time is not None:
            if isinstance(current_time, str):
                self.start_timestamp = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S").timestamp()
            else:
                self.start_timestamp = float(current_time)
        else:
            self.start_timestamp = time.time()

    def start(self):
        self.thread.start() # Bảo producer bên thread kia bắt đầu đọc video liên tục và nạp vào Queue
        return self

    def _update(self): # Vòng lặp chính của Producer Thread: Đọc liên tục từng frame từ file đĩa và nạp vào Queue cho đến khi hết video hoặc bị dừng.
        frame_counter = 0
        pushed_counter = 0

        while True:
            # Nếu cờ dừng được bật -> Thoát vòng lặp
            if self.stopped:
                break

            # Nếu Hàng đợi chưa bị đầy -> Đọc frame tiếp theo từ đĩa
            if not self.Q.full():
                grabbed, frame =self.stream.read()
        
                if not grabbed or frame is None:
                    self.stopped = True
                    break

                frame_counter += 1
                timestamp = self.start_timestamp + (frame_counter * (1.0/self.cam_fps)) # Tính toán mốc thời gian chuẩn xác của frame này theo giây (Timestamp)
                
                if (self.keep_frame_interval> 0) and (frame_counter % self.keep_frame_interval != 0):
                    continue

                self.Q.put((timestamp, pushed_counter, frame))
                pushed_counter +=1

            else: # Nếu Hàng đợi bị đầy (Queue full) -> Tạm nghỉ 1 khoảng ngắn để Consumer xử lý bớt
                time.sleep(self.fps_sleep) 

    def is_running(self):
        """
        Kiểm tra xem luồng producer - (thỏ) đọc video còn đang hoạt động hay không

        Returns:
            bool: True nếu (thỏ) vẫn đang chạy hoặc còn dữ liệu chưa đọc hết trong Queue (rổ vẫn còn carrot).
        """
        return self.has_frame() or not self.stopped

    def has_frame(self):
        """
        Kiểm tra xem còn frame nào trong Hàng đợi hay không.

        Returns:
            bool: True nếu còn frame trong Queue, False nếu đã hết và thread đã dừng.
        """
        return self.Q.qsize() > 0

    def read(self):
        """
        Hàm dành cho Consumer (Chương trình chính/AI Process):
        Lấy ra 1 frame tiếp theo từ Hàng đợi (chờ nếu hàng đợi đang trống).

        Returns:
            tuple: (timestamp, frame_index, frame_matrix)
        """
        return self.Q.get()

    def stop(self):
        """Dừng luồng đọc video và giải phóng tài nguyên Thread."""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        if hasattr(self, 'stream') and self.stream is not None and self.stream.isOpened():
            self.stream.release()

        print(f"Da stop hoan toan doc Video Stream: {self.path}")












