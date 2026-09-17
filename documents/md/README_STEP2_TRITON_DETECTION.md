# BƯỚC 2: TÍCH HỢP TRITON INFERENCE SERVER CHO PHÁT HIỆN ĐỐI TƯỢNG (YOLO DETECTION)

> **Tài liệu hướng dẫn**: Step-by-Step Triton Integration Guide  
> **Áp dụng cho thư mục**: `multi-camera-tracking-system/`  
> **Dựa trên chuẩn thiết kế**: [`OFFLINE_MULTICAM_TRACKING_GUIDE.md`](file:///D:/tmp_prj/documents/OFFLINE_MULTICAM_TRACKING_GUIDE.md#L75-L89) và [`src/main.py`](file:///d:/tmp_prj/tmp_prj/src/main.py#L58-L86)

---

## 1. Tư duy Kiến trúc & Luồng Xử lý của Bước 2

Trong mô hình xử lý đa camera, **không load model YOLO trực tiếp trong từng Process Python** (tránh tốn VRAM GPU trùng lặp). Thay vào đó, toàn bộ các Worker Process của Camera sẽ gửi ảnh qua giao thức **gRPC tốc độ cao** tới **Triton Inference Server** duy nhất chạy trên GPU.

### Sơ đồ luồng dữ liệu (Data Pipeline Flow):

```text
[ Video File / RTSP Stream ]
            │
            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   CAMERA WORKER PROCESS (Process per Cam)              │
│                                                                        │
│  1. FileVideoStream (Producer Thread) ──► Hàng đợi Queue (RAM)        │
│                                                   │                    │
│  2. AI Consumer Loop                              ▼                    │
│     ├── Gom Batch ảnh (VD: batch_size = 1 hoặc 4)                      │
│     ├── Preprocess (Resize 640x640, BGR->RGB, NCHW)                     │
│     │                                                                  │
│     ├── 3. Gọi Yolov5(ITritonClient).get_infer_result(batch_frames)    │
│     │          │                                                       │
│     │          ▼ (gRPC Request: Tensor float32 [B, 3, 640, 640])       │
│     │  ┌────────────────────────────────────────────────────────────┐  │
│     │  │       TRITON INFERENCE SERVER (GPU Server)                 │  │
│     │  │  - Model: person_det_yolov5_640 (ONNX / TensorRT)          │  │
│     │  │  - Dynamic Batching (Gom request từ nhiều Camera Process)    │  │
│     │  └──────────────────────────────┬─────────────────────────────┘  │
│     │                                 ▼ (gRPC Response: Output Tensor) │
│     │                                                                  │
│     ├── Postprocess (Sigmoid, Anchors, NMS, Scale Coords về ảnh gốc)  │
│     └── Trả về mảng danh sách [DetectionResult(box, score, 'person')]  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Cấu hình Triton Server (Server-Side Model Repository)

Triton Server yêu cầu thư mục chứa model (`model_repository`) theo đúng cấu trúc chuẩn:

```text
model_repository/
└── person_det_yolov5_640/
    ├── 1/
    │   └── model.onnx          <--- File weight ONNX đã export từ YOLOv5
    └── config.pbtxt            <--- Cấu hình Model & Dynamic Batching
```

### File cấu hình `config.pbtxt` mẫu cho YOLOv5 Person Detection:

```protobuf
name: "person_det_yolov5_640"
platform: "onnxruntime_onnx"
max_batch_size: 16

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ -1, 85 ]
  }
]

# Kích hoạt Dynamic Batching để Triton tự gom request từ nhiều Camera Process
dynamic_batching {
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

---

## 3. Cấu trúc Code Triton Client ở phía Python Backend (Client-Side)

Hệ thống mã nguồn mẫu tại `src/modules/` chia phần Triton Client thành 2 lớp rõ ràng:

### 3.1. Lớp Base `ITritonClient` (`modules/triton_client/interface.py`)
Đóng vai trò quản lý kết nối gRPC Client tới Triton Server (`host:port`), gửi `InferInput` và nhận `InferRequestedOutput`:

```python
import tritonclient.grpc as grpcclient

class ITritonClient(metaclass=ABCMeta):
    def connect(self):
        self.__triton_client = grpcclient.InferenceServerClient(
            url=f"{self.cfg['host']}:{self.cfg['port']}", verbose=False
        )
        return self.__triton_client.is_server_live() and self.__triton_client.is_model_ready(self.cfg['model_name'])

    def _infer(self, img_batch: np.ndarray):
        # Nạp dữ liệu numpy vào InferInput
        inputs = [grpcclient.InferInput(self.cfg['input_name'], img_batch.shape, "FP32")]
        inputs[0].set_data_from_numpy(img_batch)
        
        outputs = [grpcclient.InferRequestedOutput(self.cfg['output_name'])]
        
        # Gửi gRPC request sang Triton
        response = self.__triton_client.infer(
            model_name=self.cfg['model_name'],
            inputs=inputs,
            outputs=outputs
        )
        return response.as_numpy(self.cfg['output_name'])
```

### 3.2. Lớp `Yolov5` Detector (`modules/detection/yolov5/detect.py`)
Kế thừa `ITritonClient` để hiện thực 2 hàm cốt lõi:
1. `_preprocess(imgs_bgr)`: Resize ảnh về $640 \times 640$, đổi hệ màu BGR $\rightarrow$ RGB, chuẩn hóa $0..1$, chuyển về định dạng tensor NCHW (`[B, 3, 640, 640]`).
2. `_postprocess(preds)`: Chạy Sigmoid, nhân Anchors, thực thi NMS (Non-Maximum Suppression), quy đổi tọa độ box từ $640 \times 640$ về lại kích thước ảnh gốc (`img0_shape`) và lọc ra class `person`.

---

## 4. Hướng dẫn Tích hợp Chi tiết vào Code `multi-camera-tracking-system`

Dưới đây là từng bước nâng cấp từ `try_multi_stream_loader.py` lên phiên bản **đọc nhiều camera + gọi Triton YOLO Detection**.

### Bước 4.1: Định nghĩa cấu hình `triton_cfg` cho YOLO Detection

```python
triton_detection_cfg = {
    'host': 'localhost',             # IP của container Triton Server
    'port': 8001,                    # Port gRPC chuẩn của Triton (8001)
    'model_name': 'person_det_yolov5_640',
    'input_name': 'images',
    'output_name': 'output',
    'width': 640,
    'height': 640,
    'conf_thresh': 0.4,
    'iou_thresh': 0.45,
    'classes': ['person'],           # Tinh gọn: Chỉ detect class person
    'client_timeout': 10
}
```

### Bước 4.2: Tích hợp vào Worker Process đại diện cho mỗi Camera (`run_camera_worker`)

```python
def run_camera_worker(cam_info: dict, result_queue: Queue):
    cam_id = cam_info["id"]
    video_path = cam_info["path"]
    
    # 1. Khởi tạo Stream Loader cho camera này
    fvs = FileVideoStream(path=video_path, cam_fps=30)
    fvs.start()
    
    # 2. Khởi tạo Triton Detector Client NGAY TRONG PROCESS NÀY
    # (Kết nối gRPC tới Triton Server được duy trì suốt thời gian sống của Process)
    detector = Yolov5(cam_info["triton_cfg"])
    
    batch_frames = []
    batch_timestamps = []
    batch_size = 1  # Có thể gom batch_size = 2 hoặc 4 nếu cần
    
    while fvs.is_running():
        if not fvs.has_frame():
            time.sleep(0.005)
            continue
            
        timestamp, frame_idx, frame = fvs.read()
        batch_frames.append(frame)
        batch_timestamps.append(timestamp)
        
        if len(batch_frames) == batch_size:
            # 3. GỌI TRITON INFERENCE SERVER DETECT PERSON BBOX
            # Hàm get_infer_result sẽ tự chạy preprocess -> gRPC infer -> postprocess (NMS)
            batch_detection_results = detector.get_infer_result(batch_frames)
            
            for frame_mat, ts, detections in zip(batch_frames, batch_timestamps, batch_detection_results):
                # detections là danh sách các object DetectionResult(box=[x1,y1,x2,y2], score=..., classname='person')
                print(f"[{cam_id}] Frame #{frame_idx} | Detected {len(detections)} people")
                
                # -------------------------------------------------------------
                # BƯỚC TIẾP THEO (Bước 3 trong Pipeline):
                # Đưa `detections` này vào SORT Tracker để cập nhật Track ID!
                # -------------------------------------------------------------
                
            batch_frames = []
            batch_timestamps = []
            
    fvs.stop()
```

---

## 5. Mock / Fallback Client khi chưa có GPU / Triton Server

Nếu bạn đang phát triển trên laptop/PC **chưa dựng Triton Server**, bạn có thể tạo một lớp **`MockYolov5Detector`** để đóng giả lập Triton trả về BBox giả lập giúp bạn test thông luồng từ Bước 1 $\rightarrow$ Bước 2 $\rightarrow$ Bước 3 mượt mà:

```python
class MockYolov5Detector:
    """Tạo Detector giả lập không cần Triton Server cho máy không có GPU."""
    def __init__(self, cfg):
        self.cfg = cfg

    def get_infer_result(self, batch_frames):
        batch_results = []
        for frame in batch_frames:
            # Giả lập phát hiện 1 người có BBox ngẫu nhiên ở giữa màn hình
            h, w = frame.shape[:2]
            mock_box = [int(w * 0.3), int(h * 0.2), int(w * 0.6), int(h * 0.8)]
            mock_detection = DetectionResult(box=mock_box, score=0.88, classname='person')
            batch_results.append([mock_detection])
        return batch_results
```

---

## 6. Các Lưu ý Kỹ thuật Sống còn (Engineering Checklist)

1. **Chỉ lọc class `person`**:
   - Theo định hướng tinh gọn tại [`OFFLINE_MULTICAM_TRACKING_GUIDE.md:L25`](file:///D:/tmp_prj/documents/OFFLINE_MULTICAM_TRACKING_GUIDE.md#L25), loại bỏ hoàn toàn các class `hand`, `item` ở giai đoạn đầu để tăng tốc độ Inference lên **2-3 lần**.
2. **Khởi tạo gRPC Connection duy nhất 1 lần**:
   - Khởi tạo `detector = Yolov5(cfg)` ở đầu hàm `run_camera_worker`, tuyệt đối **không** tạo lại `Yolov5(cfg)` bên trong vòng lặp `while fvs.is_running()`.
3. **Cấu hình `max_queue_delay_microseconds`**:
   - Đặt thời gian chờ gom Dynamic Batch ở Triton Server là `5000` ($\mu\text{s} = 5\text{ms}$) để Triton kịp gom frame của các tiến trình camera chạy song song mà không làm trễ pipeline.

---
*Tài liệu Hướng dẫn Tích hợp Triton Detection thuộc Bước 2 trong Chuỗi phát triển Hệ thống Multi-Camera Tracking.*
