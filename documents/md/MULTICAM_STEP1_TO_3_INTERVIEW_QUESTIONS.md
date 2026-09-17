# 🎯 BỘ 40 CÂU HỎI ĐỎI ĐÁP ĐẦU THỦ & PHỎNG VẤN ĐỦ BỘ (STEP 1 - STEP 3)
## HỆ THỐNG MULTI-CAMERA OFFLINE TRACKING & TRACKLET PACKAGING

> **Dành cho**: Kỹ sư AI / Computer Vision Intern & Junior  
> **Vai trò Mentor**: Technical Lead / System Architect  
> **Mục đích**: Giúp lập trình viên thấu hiểu từ bản chất (First Principles) lý do thiết kế mã nguồn, trả lời phỏng vấn xuất sắc và bảo vệ đồ án/dự án trước Hội đồng Kỹ thuật.

---

## 📊 THỐNG KÊ TOÀN BỘ TIẾN TRÌNH (PROCESSES) & LUỒNG (THREADS) TRONG CODEBASE

$$\text{TỔNG CỘNG HỆ THỐNG RUNTIME}: \mathbf{5\text{ Tiến trình (Processes)}} + \mathbf{3\text{ Luồng ngầm (Threads)}}$$

### 🔴 1. DANH SÁCH 5 TIẾN TRÌNH (PROCESSES)
*(Mỗi Process có PID riêng, RAM riêng, chìa khóa GIL riêng do OS cấp)*

1. **Process 1: Master (Main Process)**
   - **Được tạo từ**: Lệnh `python test/try_step3_tracking_and_packaging.py` (hàm `main()`).
   - **Nhiệm vụ**: Điều phối Queues, gom frame 3 camera ghép nằm ngang và nén ghi ra tệp `.mp4`.
2. **Process 2: AI Detection Server**
   - **Được tạo từ**: `PythonMiniTritonServer` trong [`modules/detection/python_mini_triton.py`](file:///home/ngocnm/multi-camera-tracking-system/modules/detection/python_mini_triton.py).
   - **Nhiệm vụ**: Nạp model YOLOv8 lên GPU 5 (`cuda:5`), gom batch và tính toán Bounding Box.
3. **Process 3: Camera Worker 1 (CAM_0351)**
   - **Được tạo từ**: `Process(target=run_camera_worker_step3, args=(CAM_0351,...))` trong [`try_step3_tracking_and_packaging.py`](file:///home/ngocnm/multi-camera-tracking-system/test/try_step3_tracking_and_packaging.py).
   - **Nhiệm vụ**: Chạy SORT Tracker, đệm RAM `dict_tracks` và đóng gói JSON cho Cam 351.
4. **Process 4: Camera Worker 2 (CAM_0353)**
   - **Được tạo từ**: `Process(target=run_camera_worker_step3, args=(CAM_0353,...))`.
   - **Nhiệm vụ**: Chạy SORT Tracker, đệm RAM `dict_tracks` và đóng gói JSON cho Cam 353.
5. **Process 5: Camera Worker 3 (CAM_0358)**
   - **Được tạo từ**: `Process(target=run_camera_worker_step3, args=(CAM_0358,...))`.
   - **Nhiệm vụ**: Chạy SORT Tracker, đệm RAM `dict_tracks` và đóng gói JSON cho Cam 358.

---

### 🔵 2. DANH SÁCH 3 LUỒNG NGẦM (THREADS)
*(Mỗi Thread nằm bên trong không gian bộ nhớ của từng Camera Worker Process)*

Bên trong mỗi Camera Worker Process, khi gọi `FileVideoStream` ([`modules/loader/try_file_stream.py`](file:///home/ngocnm/multi-camera-tracking-system/modules/loader/try_file_stream.py)), một Luồng đọc đĩa ngầm được khởi tạo:

1. **Thread 1 (Video Reader Thread Cam 351)**:
   - **Nằm trong**: Process 3 (Cam 351 Worker).
   - **Được tạo từ**: `threading.Thread(target=self.update)` trong `FileVideoStream`.
   - **Nhiệm vụ**: Gọi `cv2.VideoCapture.read()` đọc đĩa video Cam 351 nạp trước frame vào Queue ngầm để Process 3 không bị nghẽn đĩa.
2. **Thread 2 (Video Reader Thread Cam 353)**:
   - **Nằm trong**: Process 4 (Cam 353 Worker).
   - **Nhiệm vụ**: Đọc đĩa video Cam 353 nạp trước frame vào Queue.
3. **Thread 3 (Video Reader Thread Cam 358)**:
   - **Nằm trong**: Process 5 (Cam 358 Worker).
   - **Nhiệm vụ**: Đọc đĩa video Cam 358 nạp trước frame vào Queue.

> **🎯 TẠI SAO LẠI DÙNG THREAD Ở BƯỚC ĐỌC VIDEO MÀ KHÔNG DÙNG PROCESS?**  
> Vì việc đọc đĩa video (`VideoCapture.read()`) là tác vụ **I/O Bound (chờ ổ đĩa đọc dữ liệu)**. Trong lúc chờ đĩa đọc, Python sẽ tự động giải phóng khóa GIL cho luồng đó $\implies$ Dùng Thread ở bước đọc đĩa là cực kỳ tiết kiệm bộ nhớ RAM mà vẫn đạt hiệu năng $100\%$!

---

## 🏛️ PHẦN 1: TỔNG QUAN KIẾN TRÚC HỆ THỐNG & ĐA TIẾN TRÌNH (SYSTEM ARCHITECTURE & MULTIPROCESSING)

### Câu 1: Tại sao hệ thống này lại phải chia thành 5 tiến trình (Process) riêng biệt thay vì dùng 1 tiến trình duy nhất chạy tất cả?
* **Trả lời chuẩn Mentor (First Principles)**:
  1. **Bản chất phần cứng CPU**: CPU có nhiều nhân vật lý. Xử lý 3 luồng video $1080\text{p}$ đồng thời bao gồm: Giải mã khung hình + Kalman Filter + Vẽ đồ họa. Phải chia lên nhiều Nhân CPU mới chạy song song được.
  2. **Rào cản Khóa GIL (Global Interpreter Lock)**: Trong CPython, khóa GIL chỉ cho phép duy nhất 1 Thread được thực thi mã bytecode Python tại 1 thời điểm. Nếu dùng 1 tiến trình duy nhất, 3 Camera + AI Server + Master sẽ tranh giành 1 chiếc chìa khóa GIL $\implies$ Tốc độ bị kéo tụt xuống 3-5 FPS.
  3. **Giải pháp Đa tiến trình (`multiprocessing.Process`)**: Mỗi `Process` được OS cấp 1 bộ nhớ ảo riêng, 1 Trình thông dịch riêng và **1 chiếc chìa khóa GIL riêng**. 5 Tiến trình = 5 chìa khóa GIL $\implies$ Linux Kernel phân bổ lên 5 Nhân CPU chạy song song thật sự $100\%$.
  4. **Số lượng 5**: Đã được tính toán tối ưu vừa khít (1 Master Orchestrator + 1 AI GPU Server + 3 Camera Workers).

### Câu 2: Tại sao AI Detection Server (`PythonMiniTritonServer`) lại được tách thành 1 tiến trình riêng độc lập với 3 tiến trình Camera Worker và Main Process?
* **Trả lời chuẩn Mentor (First Principles)**:
  1. **Bản chất GPU & Parallel Math**: GPU xử lý Tensor 4D $[B, C, H, W]$. Thời gian GPU nhân ma trận với Batch Size $B=3$ gần như tương đương thời gian chạy $B=1$.
  2. **Tối ưu Dynamic Batching Throughput**: Nếu mỗi Cam tự nạp model riêng ($B=1$), GPU phải làm việc 3 lần ($8\text{ms} \times 3 = 24\text{ms}$). Tách AI Server tập trung giúp gom 3 ảnh gửi tới làm 1 Batch ($B=3$) cho GPU tính 1 lần duy nhất ($\sim 9\text{ms}$ cho CẢ 3 CAM), tăng tốc độ gấp gần 3 lần.
  3. **Xung đột GIL với Main Process**: Main Process bận nén video và ghi đĩa MP4 trên CPU. Nếu gộp AI Server vào Main Process, khi Main Process bận ghi đĩa sẽ giữ chìa khóa GIL $\implies$ AI Server không gọi được GPU $\implies$ GPU và 3 Camera Workers đứng hình chờ. Tách 2 tiến trình riêng giúp AI Server có GIL riêng, trả BBox liên tục $5\text{ms}$ không bao giờ bị nghẽn!

### Câu 3: Các tiến trình giao tiếp và truyền dữ liệu frame/bbox với nhau bằng cơ chế nào?
* **Trả lời chuẩn Mentor**: Sử dụng `multiprocessing.Queue` (hàng đợi IPC trong bộ nhớ shared memory). Camera Worker đẩy frame thô vào `request_queue`, AI Server đọc từ `request_queue`, tính toán BBox trên GPU và trả kết quả qua `response_queue[cam_id]`.

### Câu 4: Tại sao không dùng Shared Memory Mmap để truyền ảnh giữa các tiến trình mà lại dùng Queue?
* **Trả lời chuẩn Mentor**: `Queue` trong Python đã được tối ưu sẵn bằng Pickle/Pipe cho IPC. Với số lượng 3 camera ở độ phân giải $480 \times 270$ để visualization, băng thông của `Queue` dư sức đáp ứng $30\text{ FPS}$ mà code lại đơn giản, an toàn, tránh lỗi tranh chấp bộ nhớ (race condition).

### Câu 5: Sự khác biệt giữa `Multiprocessing` và `Multithreading` trong Python là gì và bài toán này dùng cái nào ở đâu?
* **Trả lời chuẩn Mentor**:
  - `Multiprocessing`: Tạo không gian bộ nhớ riêng, vượt GIL. Dùng cho các tiến trình chính: AI Server, các Camera Worker.
  - `Multithreading`: Dùng chung bộ nhớ, bị GIL hạn chế. Dùng bên trong `FileVideoStream` chỉ để làm nhiệm vụ I/O đọc khung hình video từ đĩa vào RAM (I/O bound).

### Câu 6: Làm thế nào để đảm bảo khi ấn ngắt Ctrl+C hoặc khi xong video, tất cả 5 tiến trình tự đóng sạch sẽ không bị treo ngầm trên GPU?
* **Trả lời chuẩn Mentor**:
  1. Sử dụng sự kiện `ready_event.wait()` để đồng bộ khởi động.
  2. Bắt tín hiệu kết thúc `vis_queue.put((cam_id, -1, None))`.
  3. Tiến trình Master gọi `worker.join()` chờ các con xả hết dữ liệu rồi mới gọi `ai_server_process.terminate()`.

### Câu 7: Điều gì xảy ra nếu 1 Camera Worker chạy chậm hơn 2 camera còn lại?
* **Trả lời chuẩn Mentor**: Hệ thống tự động cân bằng nhờ cơ chế **Asynchronous Queue**. Các Worker độc lập tự đọc và xử lý theo tốc độ riêng. AI Server sẽ gom những frame đang có sẵn trong `request_queue` tại thời điểm đó (dynamic batch size từ 1 đến max_batch_size).

---

## 📹 PHẦN 2: MODULE ĐỌC VIDEO LOADER (`modules/loader/`)

### Câu 8: Lớp `FileVideoStream` giải quyết vấn đề gì so với việc gọi `cv2.VideoCapture.read()` trực tiếp trong vòng lặp chính?
* **Trả lời chuẩn Mentor**: `cv2.VideoCapture.read()` là thao tác I/O đồng bộ (blocking). Nếu gọi trực tiếp trong vòng lặp chính, tiến trình phải dừng lại chờ đĩa đọc xong frame rồi mới xử lý tiếp. `FileVideoStream` chạy 1 background thread đọc trước các frame nạp vào `queue.Queue`, giúp vòng lặp chính luôn có sẵn frame để dùng ngay ($0\text{ms}$ nghẽn).

### Câu 9: Tại sao trong `FileVideoStream` phải giới hạn `queue_size=20` mà không để hàng đợi phình to tự do?
* **Trả lời chuẩn Mentor**: Để **kiểm soát RAM**. Nếu video dài hàng ngàn frame và không giới hạn Queue, thread đọc đĩa sẽ nạp toàn bộ video vào RAM gây ra tràn bộ nhớ (Out of Memory - OOM). `queue_size=20` đảm bảo đệm đủ mượt mà chỉ tốn vài chục MB RAM.

### Câu 10: Làm sao để đồng bộ mốc thời gian thực (Timestamp) giữa các video quay độc lập?
* **Trả lời chuẩn Mentor**: `FileVideoStream` tính toán timestamp dựa trên công thức: `timestamp = start_time + (frame_idx / cam_fps)`. Điều này biến chỉ số khung hình thành mốc thời gian dạng epoch/datetime chuẩn để so khớp liên camera ở Step 4.

### Câu 11: Backend "FFmpeg" khác gì backend "PyAV" hay "OpenCV" mặc định trong `FileVideoStream`?
* **Trả lời chuẩn Mentor**: FFmpeg C-library cho khả năng seek khung hình chính xác theo timestamp và giải mã phần cứng (Hardware Accelerated H.264/H.265) mượt mà hơn OpenCV mặc định trên các hệ thống Linux Server.

### Câu 12: Điều gì xảy ra khi luồng đọc video chạm mốc EOF (End of File)?
* **Trả lời chuẩn Mentor**: Cờ `self.stopped` được bật thành `True`, thread đọc dừng lại, và hàm `.read()` sẽ trả về cờ báo hết frame để tiến trình Worker biết và chuyển sang giai đoạn xả bộ nhớ RAM.

---

## 🤖 PHẦN 3: AI DETECTION ENGINE (`modules/detection/`)

### Câu 13: Tại sao lại chọn YOLOv8 cho bước phát hiện người mà không dùng YOLOv5 hay Faster R-CNN?
* **Trả lời chuẩn Mentor**: YOLOv8 sử dụng kiến trúc Anchor-free kèm C2f module, cho độ chính xác mAP cao hơn YOLOv5 trên cùng tốc độ inference, và nhanh gấp nhiều lần Faster R-CNN, đáp ứng chuẩn xử lý thời gian thực 30 FPS.

### Câu 14: Tại sao trong code Detection chỉ lọc duy nhất class `person` (người) và bỏ qua tất cả các class khác như `car`, `chair`?
* **Trả lời chuẩn Mentor**: Dự án tập trung giải bài toán Multi-Camera Person Tracking. Lọc bỏ các class thừa ngay tại đầu ra YOLO giúp giảm kích thước mảng dữ liệu truyền qua IPC Queue và giảm khối lượng tính toán cho tracker.

### Câu 15: Tại sao lại gọi là `PythonMiniTriton`? Nó mô phỏng lại điều gì của Nvidia Triton Inference Server?
* **Trả lời chuẩn Mentor**: Mô phỏng lại 2 tính năng cốt lõi của Nvidia Triton:
  1. **Dynamic Batching**: Tự động dồn các request lẻ thành 1 Batch lớn trước khi cho GPU tính toán.
  2. **Model Decoupling**: Tách rời tiến trình AI Inference khỏi tiến trình Logic ứng dụng qua giao thức Request/Response.

### Câu 16: Biến `max_delay_sec=0.005` trong AI Server có ý nghĩa gì?
* **Trả lời chuẩn Mentor**: Là thời gian tối đa AI Server chấp nhận "đứng chờ" để gom đủ `max_batch_size`. Nếu sau $5\text{ms}$ mà chưa gom đủ 8 frame, nó vẫn lập tức đẩy số frame hiện có vào GPU để tránh làm tăng độ trễ (latency) của hệ thống.

### Câu 17: Làm thế nào để chạy AI Server trên 1 GPU cụ thể (ví dụ GPU 5) trong máy có 8 GPU?
* **Trả lời chuẩn Mentor**: 
  - Trong code Python: Truyền `device="cuda:5"` khi khởi tạo PyTorch model `YOLO("yolov8n.pt").to(device)`.
  - Trên lệnh Shell: Đặt biến môi trường `CUDA_VISIBLE_DEVICES=5 python ...`.

### Câu 18: Tại sao lại trả về đối tượng `DetectionResult` chứa `box=[x1,y1,x2,y2]` dạng số nguyên (integer) thay vì dạng float chuẩn hóa $[0, 1]$?
* **Trả lời chuẩn Mentor**: Tọa độ pixel dạng số nguyên $[x_1, y_1, x_2, y_2]$ giúp phép cắt ảnh người `frame[y1:y2, x1:x2]` bằng OpenCV diễn ra ngay lập tức mà không phải tốn phép nhân chuyển đổi kích thước ảnh ở các bước sau.

---

## 🎯 PHẦN 4: SINGLE-CAM TRACKING & SORT ALGORITHM (`modules/tracking/`)

### Câu 19: Thuật toán SORT (Simple Online and Realtime Tracking) gồm 2 thành phần toán học chính nào?
* **Trả lời chuẩn Mentor**:
  1. **Kalman Filter**: Dự đoán vị trí Bounding Box ở khung hình tiếp theo dựa trên vận tốc di chuyển.
  2. **Hungarian Algorithm**: Ghép nối (association) giữa ô BBox dự đoán và ô BBox thực tế nhận được từ YOLO dựa trên ma trận IoU.

### Câu 20: Không gian trạng thái (State Space) 7 chiều $[x, y, s, r, \dot{x}, \dot{y}, \dot{s}]$ trong `KalmanBoxTracker` đại diện cho những đại lượng nào?
* **Trả lời chuẩn Mentor**:
  - $x, y$: Tọa độ tâm của Bounding Box.
  - $s$: Diện tích ô vuông ($\text{scale} = \text{width} \times \text{height}$).
  - $r$: Tỷ lệ khung hình ($\text{aspect ratio} = \text{width} / \text{height}$).
  - $\dot{x}, \dot{y}, \dot{s}$: Vận tốc thay đổi tương ứng của tâm $x, y$ và diện tích $s$.

### Câu 21: Tại sao tỷ lệ $r$ (aspect ratio) lại được coi là hằng số không có vận tốc $\dot{r}$ trong mô hình Kalman Filter?
* **Trả lời chuẩn Mentor**: Vì cơ thể người khi bước đi không thay đổi đột ngột tỷ lệ chiều cao/chiều rộng. Việc coi $r$ là hằng số giúp mô hình dự đoán Kalman Filter ổn định, không bị méo ô BBox khi gặp nhiễu.

### Câu 22: Phép tính IoU (Intersection over Union) đóng vai trò gì trong hàm `associate_detections_to_trackers`?
* **Trả lời chuẩn Mentor**: Đóng vai trò là **Hàm chi phí (Cost Matrix)**. IoU đo khoảng không gian đè lên nhau giữa BBox thực tế và BBox dự đoán. IoU càng cao ($1.0$) nghĩa là 2 ô BBox càng trùng khớp.

### Câu 23: Tại sao lại dùng `linear_sum_assignment(-iou_matrix)` với dấu âm `-`?
* **Trả lời chuẩn Mentor**: Thuật toán Hungarian trong SciPy (`linear_sum_assignment`) mặc định bài toán tìm **chi phí nhỏ nhất (Minimization)**. Đặt dấu âm `-iou_matrix` biến bài toán thành **tìm tổng IoU lớn nhất (Maximization)**.

### Câu 24: Tham số `low_iou_threshold=0.25` giải quyết vấn đề gì?
* **Trả lời chuẩn Mentor**: Lọc bỏ các cặp ghép lỗi có IoU $< 0.25$. Nếu một BBox dự đoán và BBox thực tế ở quá xa nhau (IoU $< 0.25$), hệ thống sẽ không gán ID cũ mà coi đó là một người mới xuất hiện hoặc người cũ đã bị che khuất.

### Câu 25: Tham số `min_hits=3` và `max_age=30` trong `PersonTracker` có ý nghĩa gì?
* **Trả lời chuẩn Mentor**:
  - `min_hits=3`: Một người mới xuất hiện phải được phát hiện liên tiếp ít nhất 3 frame mới được xác nhận gán Track ID chính thức (tránh nhiễu phát hiện ảo của YOLO).
  - `max_age=30`: Nếu người đó bị tạm che khuất trong tối đa 30 frame (1 giây), tracker vẫn giữ ID chờ người đó xuất hiện lại. Nếu quá 30 frame không thấy $\rightarrow$ xóa tracker.

### Câu 26: Tại sao mỗi ID người lại được cấp 1 màu sắc riêng biệt ngẫu nhiên (`get_color_for_id`) khi vẽ lên video?
* **Trả lời chuẩn Mentor**: Dựa trên hàm băm (hashing) `seed = track_id * 31 + 7`. Giúp Kỹ sư kiểm thử bằng mắt thường (Visual Inspection) nhận biết ngay lập tức nếu xảy ra hiện tượng nhảy ID (ID Jumping/Switching).

### Câu 27: Điều gì xảy ra khi 2 người đi cắt ngang qua nhau (Occlusion)? SORT giải quyết ra sao?
* **Trả lời chuẩn Mentor**: Kalman Filter tiếp tục dự đoán vị trí 2 ô BBox theo vận tốc cũ. Khi 2 ô tách ra, Hungarian IoU Match sẽ nối lại ID đúng nếu thời gian che khuất nhỏ hơn `max_age`.

---

## 🧠 PHẦN 5: QUẢN LÝ BỘ NHỚ RAM ĐỆM & CROP SELECTION (`modules/tracking/track_manager.py`)

### Câu 28: Lớp `PersonTrackManager` có vai trò gì và lưu trữ dữ liệu ở đâu?
* **Trả lời chuẩn Mentor**: Đóng vai trò là **RAM Buffer Manager (Bộ nhớ đệm trong RAM)**. Nó lưu trữ từ điển `dict_tracks` chứa lịch sử tọa độ BBox, thời gian vào/ra và danh sách các bức ảnh cắt nét (`cropped_imgs`) của từng người đang xuất hiện.

### Câu 29: Hàm `is_good_box()` lọc ảnh crop dựa trên những tiêu chí vật lý nào?
* **Trả lời chuẩn Mentor**:
  1. Kích thước tối thiểu: Chiều rộng và chiều cao $> 15\text{px}$ (bỏ ảnh quá nhỏ/mờ).
  2. Tỷ lệ hình thể: Chiều cao / Chiều rộng $\ge 0.5$ (đảm bảo là dáng người đứng/đi, không phải ảnh cắt lỗi nằm ngang).

### Câu 30: Tại sao không lưu TOÀN BỘ ảnh crop của tất cả các frame vào RAM mà phải qua `is_good_box()`?
* **Trả lời chuẩn Mentor**: Để **bảo vệ bộ nhớ RAM**. Một người ở lại camera 10 phút sinh ra 18.000 frame ảnh crop. Lưu toàn bộ sẽ làm tràn RAM (OOM) chỉ sau vài phút. `is_good_box()` giúp chỉ giữ lại vài chục ảnh rõ nét nhất.

### Câu 31: Khái niệm `real_dead_tracks` là gì và được xác định theo công thức toán học nào?
* **Trả lời chuẩn Mentor**: Là trạng thái xác nhận một người đã **thực sự rời khỏi góc quay camera**.  
  Công thức: `current_timestamp - track_info.dead_time > max_fragment_time` (với `max_fragment_time = 1.0` đến `3.0` giây).

### Câu 32: Tại sao phải chờ thêm `max_fragment_time` (ví dụ 1.0 giây) sau khi tracker báo chết mới xác nhận là `real_dead_track`?
* **Trả lời chuẩn Mentor**: Để đề phòng người đó chỉ tạm thời bị che khuất đằng sau cột nhà hoặc góc khuất ngắn. Chờ 1.0 giây đảm bảo người đó đã thực sự đi ra khỏi tầm quay của camera.

### Câu 33: Hàm `remove_track(track_id)` gọi khi nào và đóng vai trò sinh tử gì trong quản lý bộ nhớ?
* **Trả lời chuẩn Mentor**: Được gọi ngay lập tức sau khi dữ liệu của `track_id` đó đã được đóng gói và lưu ra file JSON. Nó xóa chìa khóa `track_id` khỏi `dict_tracks`, giải phóng $100\%$ dung lượng RAM của người đó, giúp RAM máy tính giữ ở mức ổn định $< 500\text{MB}$ suốt quá trình chạy.

---

## 💎 PHẦN 6: FASTREID VECTORIZER, CLUSTERING & TRACKLET PACKAGING

### Câu 34: Vector đặc trưng 256 chiều (256-dim embedding) đại diện cho thông tin gì của đối tượng?
* **Trả lời chuẩn Mentor**: Đại diện cho **đặc trưng diện mạo (Appearance Feature)** của người đó (màu áo, màu quần, kiểu tóc, dáng người) được nén thành 256 con số thực đã chuẩn hóa độ dài $L_2$ (`norm = 1.0`).

### Câu 35: Tại sao trong `FastReIDVectorizer` lại áp dụng chuẩn hóa $L_2$ (`L2 Normalization`) cho vector?
* **Trả lời chuẩn Mentor**: Chuẩn hóa $L_2$ đưa mọi vector về độ dài bằng $1.0$. Khi đó, phép tính **Cosine Distance** giữa 2 vector chỉ đơn giản là tích vô hướng (Dot Product) `np.dot(vec1, vec2)`, giúp tốc độ so sánh ở Step 4 nhanh gấp hàng chục lần.

### Câu 36: Tại sao lại cần thuật toán Gom nhóm `TrackletClusterer` (MiniBatchKMeans) trước khi đóng gói Tracklet?
* **Trả lời chuẩn Mentor**: Trong 300 frame, một người có thể sinh ra 100 ảnh crop (nhiều ảnh bị trùng góc nhìn hoặc bị nhòe chuyển động). `MiniBatchKMeans` gom 100 vector đó thành 5 cụm (Clusters) và chọn ra **5 vector tâm (Medoids) xuất sắc nhất**, giúp nén dung lượng dữ liệu giảm 20 lần mà vẫn giữ nguyên đặc trưng nét nhất.

### Câu 37: Cấu trúc của một tệp gói Tracklet JSON (`TrackletPackage`) gồm những trường dữ liệu cốt lõi nào?
* **Trả lời chuẩn Mentor**:
  - `cam_id`: Mã camera (VD: `"CAM_0351"`).
  - `track_id`: ID tạm thời ở camera đó (VD: `1`).
  - `start_time` & `end_time`: Mốc thời gian bước vào và rời khỏi camera.
  - `boxes`: Danh sách tọa độ BBox theo thời gian.
  - `rep_vectors`: Mảng chứa Top 5 vector đặc trưng 256 chiều.

### Câu 38: Sự khác biệt giữa việc lưu gói Tracklet ra tệp JSON và lưu vào CSDL MongoDB `TrackResult` là gì?
* **Trả lời chuẩn Mentor**:
  - **Tệp JSON**: Phù hợp chạy Offline trên máy đơn, đọc/ghi đĩa trực tiếp, không tốn tài nguyên cài đặt DB.
  - **MongoDB `TrackResult`**: Phù hợp hệ thống Production phân tán nhiều Server, hỗ trợ đánh Index và truy vấn theo mốc giờ `end_time` cực nhanh.

### Câu 39: Tại sao toàn bộ bước FastReID Vectorizer và Clustering hiện tại lại cho chạy trên CPU mà không đẩy lên GPU?
* **Trả lời chuẩn Mentor**: Để **chia lửa tài nguyên**. GPU 5 được dành riêng 100% công suất cho YOLOv8 Detector để đạt $30\text{ FPS}$. Việc chạy ReID/Clustering trên CPU tận dụng các nhân CPU rảnh rỗi, tránh nghẽn VRAM và tranh chấp luồng GPU.

### Câu 40: Dữ liệu tệp JSON Tracklet thu được ở Step 3 sẽ được truyền tiếp sang Step 4 (Multi-Camera ReID) như thế nào?
* **Trả lời chuẩn Mentor**: Ở Step 4, ReID Engine sẽ quét toàn bộ thư mục `output_tracklets/*.json`, đọc tất cả gói Tracklet vào mảng Python, **sắp xếp tăng dần theo `end_time`**, và tính khoảng cách Cosine Distance giữa các `rep_vectors` của các camera khác nhau để gán **Global Customer ID (`c_id`)**.

---
*Bộ 40 câu hỏi bảo vệ & phỏng vấn thuộc Chuỗi phát triển Hệ thống Multi-Camera Offline Tracking (Step 1 -> Step 3).*
