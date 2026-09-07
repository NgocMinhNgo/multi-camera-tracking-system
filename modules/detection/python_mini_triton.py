"""
PYTHON MINI-TRITON INFERENCE SERVER (PURE PYTHON MULTIPROCESSING PATTERN)
--------------------------------------------------------------------------------
Mục đích:
1. Mô phỏng 90% kiến trúc của NVIDIA Triton Inference Server bằng 100% Python thuần.
2. Nạp Model AI DUY NHẤT 1 LẦN trong RAM/VRAM GPU (Tiết kiệm VRAM).
3. Hỗ trợ Dynamic Batching: Tự động gom các frame lẻ từ nhiều Camera Process trong 5ms thành 1 Batch trước khi chạy GPU.
4. Không cần Docker, không cần Triton Server, không cần quyền admin sudo.

Dựa trên chuẩn thiết kế trong:
- OFFLINE_MULTICAM_TRACKING_GUIDE.md (Công đoạn 2)
- src/main.py (Central Inference Engine Architecture)
"""

import time
import os
import sys
import cv2
import numpy as np
from multiprocessing import Process, Queue


class DetectionResult:
    """Đơn vị lưu trữ kết quả Bounding Box phát hiện đối tượng."""
    def __init__(self, box, score, classname='person'):
        self.box = box            # [x1, y1, x2, y2]
        self.score = score        # Confidence score (0..1)
        self.classname = classname

    def __repr__(self):
        return f"DetectionResult(box={self.box}, score={self.score:.2f}, class={self.classname})"


class MockDetectorEngine:
    """Engine phát hiện đối tượng giả lập (Fall-back khi chưa cài PyTorch/Ultralytics)."""
    def __init__(self):
        print("[Python-Mini-Triton] Initialized Mock Detector Engine (Lightweight OpenCV)...")

    def infer_batch(self, batch_frames):
        batch_results = []
        for frame in batch_frames:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detections = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 1000:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    detections.append(DetectionResult(box=[x, y, x + bw, y + bh], score=0.95, classname='person'))
            batch_results.append(detections)
        return batch_results


class RealYoloDetectorEngine:
    """Engine phát hiện đối tượng thực tế bằng Ultralytics YOLOv8."""
    def __init__(self, model_path="yolov8n.pt", device="cpu", conf_thresh=0.35):
        from ultralytics import YOLO
        print(f"[Python-Mini-Triton] Loading REAL YOLOv8 Model ('{model_path}') on device: '{device}'...")
        self.model = YOLO(model_path)
        self.device = device
        self.conf_thresh = conf_thresh

    def infer_batch(self, batch_frames):
        results = self.model(batch_frames, conf=self.conf_thresh, device=self.device, verbose=False)
        batch_results = []
        for res in results:
            detections = []
            if res.boxes is not None:
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    classname = self.model.names[cls_id]
                    # Chỉ lọc class 'person'
                    if classname == 'person':
                        xyxy = box.xyxy[0].tolist()
                        score = float(box.conf[0])
                        b = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        detections.append(DetectionResult(box=b, score=score, classname='person'))
            batch_results.append(detections)
        return batch_results


class PythonMiniTritonServer(Process):
    """
    Tiến trình AI Trung tâm (Tương đương Triton Inference Server).
    Chỉ chạy 1 Instance duy nhất trên GPU/CPU, quản lý Dynamic Batching.
    """
    def __init__(self, request_queue: Queue, response_queues: dict, max_batch_size=8, max_delay_sec=0.005, use_real_yolo=True):
        super().__init__()
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.max_batch_size = max_batch_size
        self.max_delay_sec = max_delay_sec
        self.use_real_yolo = use_real_yolo
        self.daemon = True

    def run(self):
        print(f"[Python-Mini-Triton] Central Server Engine Started (PID: {os.getpid()})...")
        
        # Thử nạp PyTorch / Ultralytics YOLOv8, nếu không có thì dùng Mock Engine
        if self.use_real_yolo:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                detector_engine = RealYoloDetectorEngine(model_path="yolov8n.pt", device=device, conf_thresh=0.35)
            except Exception as e:
                print(f"[Python-Mini-Triton] Warning: Could not load Real YOLO ({e}). Falling back to Mock Engine.")
                detector_engine = MockDetectorEngine()
        else:
            detector_engine = MockDetectorEngine()


        while True:

            batch_frames = []
            batch_requests = []

            start_wait = time.time()
            
            # DYNAMIC BATCHING: Chờ tối đa 5ms để gom frame từ các Camera Worker khác nhau
            while (time.time() - start_wait < self.max_delay_sec) and (len(batch_frames) < self.max_batch_size):
                if not self.request_queue.empty():
                    req = self.request_queue.get()
                    if req == "STOP":
                        print("[Python-Mini-Triton] Received STOP signal. Shutting down Central Server.")
                        return
                    
                    cam_id, frame_idx, frame = req
                    batch_frames.append(frame)
                    batch_requests.append((cam_id, frame_idx))
                else:
                    time.sleep(0.0005)

            # Nếu gom được ít nhất 1 frame -> Đẩy vào GPU/Engine chạy 1 lượt!
            if len(batch_frames) > 0:
                batch_detections = detector_engine.infer_batch(batch_frames)

                # Phân phát kết quả BBox về đúng Response Queue của từng Camera
                for (cam_id, frame_idx), detections in zip(batch_requests, batch_detections):
                    if cam_id in self.response_queues:
                        self.response_queues[cam_id].put((frame_idx, detections))


class PythonMiniTritonClient:
    """
    Client giao tiếp với PythonMiniTritonServer dành cho từng Camera Worker Process.
    """
    def __init__(self, cam_id: str, request_queue: Queue, response_queue: Queue):
        self.cam_id = cam_id
        self.request_queue = request_queue
        self.response_queue = response_queue

    def detect(self, frame_idx: int, frame: np.ndarray):
        """Gửi frame sang Central AI Process và chờ nhận kết quả BBox trả về."""
        # 1. Gửi request sang Central Server Queue
        self.request_queue.put((self.cam_id, frame_idx, frame))
        
        # 2. Chờ nhận kết quả từ Response Queue của camera này
        res_frame_idx, detections = self.response_queue.get()
        return detections
