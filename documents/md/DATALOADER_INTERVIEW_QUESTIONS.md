# BỘ 30 CÂU HỎI PHỎNG VẤN CHUYÊN SÂU: VIDEO DATALOADER & STREAMING ARCHITECTURE

> **Dành cho**: Kỹ sư Phần mềm / Kỹ sư AI muốn tự kiểm tra và làm chủ module `FileVideoStream`  
> **Người soạn**: Senior Tech Lead & Principal System Architect  
> **Phạm vi**: **Video Stream Loading, Producer-Consumer Pattern, Multi-threading, Thread-Safe Queue & SOLID Design Principles**.

---

## MỤC LỤC BỘ CÂU HỎI

* **Phần 1: Kiến trúc & Triết lý Thiết kế (6 câu)**
* **Phần 2: Mô hình Producer-Consumer & Threading (6 câu)**
* **Phần 3: Hàng đợi Đệm RAM Queue & An toàn Bộ nhớ (6 câu)**
* **Phần 4: Xử lý FPS, Timestamp & Lọc Khung hình (6 câu)**
* **Phần 5: Quản lý Vòng đời & Xử lý Lỗi (6 câu)**

---

## PHẦN 1: KIẾN TRÚC & TRIẾT LÝ THIẾT KẾ (ARCHITECTURAL MINDSET)

### Câu 1: Tại sao lại cần tạo class `FileVideoStream` đóng gói OpenCV và Thread thay vì gọi `cv2.VideoCapture` trực tiếp ở file `main.py`?
* **Gợi ý trả lời của Tech Lead**:
  * **Tách biệt Trách nhiệm (Encapsulation / SRP)**: File `main.py` chỉ làm nhiệm vụ cao cấp là chạy mô hình AI (YOLOv5 / Tracker). Việc đọc đĩa I/O, tạo Thread, nạp Queue là "việc vặt" cần được bọc kín trong `FileVideoStream`.
  * **Tránh rác code khi mở rộng**: Nếu có 10 camera, việc gọi `cv2.VideoCapture` trực tiếp sẽ làm `main.py` phình to ra hàng trăm dòng code quản lý đĩa cứng, rất khó bảo trì và dễ sinh ra bug.

### Câu 2: Class `FileVideoStream` thể hiện Nguyên lý Đơn trách nhiệm (Single Responsibility Principle - chữ S trong SOLID) như thế nào?
* **Gợi ý trả lời của Tech Lead**:
  * Class chỉ có **ĐÚNG MỘT LÝ DO DUY NHẤT ĐỂ THAY ĐỔI**: Khi cách thức nạp và đệm khung hình video bị thay đổi.
  * Nó không chứa logic AI, không chứa logic Tracker, không chứa logic lưu Database.

### Câu 3: Tại sao không gọi trực tiếp `fvs.Q.get()` ở file chính mà lại phải thông qua hàm public `fvs.read()`?
* **Gợi ý trả lời của Tech Lead**:
  * **Che giấu Chi tiết cài đặt (OOP Encapsulation)**: `self.Q` là thuộc tính nội bộ. Bên ngoài không cần biết bên trong dùng `queue.Queue`, `Redis`, hay `Kafka`.
  * **An toàn & Dễ mở rộng**: Sau này nếu muốn thay `Queue` bằng một cơ chế khác, ta chỉ cần sửa bên trong hàm `read()`, tất cả các file bên ngoài gọi `fvs.read()` không bị văng lỗi và không cần sửa 1 dòng code nào.

### Câu 4: Sự khác biệt về mặt bản chất giữa tên khái niệm `DataLoader` và tên cụ thể `FileVideoStream` là gì?
* **Gợi ý trả lời of Tech Lead**:
  * `DataLoader` là tên danh từ chung chỉ bất kỳ class nào nạp dữ liệu vào app.
  * `FileVideoStream` là tên gọi cụ thể hoá, mô tả chính xác nguồn dữ liệu (tệp đĩa `File`) và dạng dữ liệu (luồng ảnh `VideoStream`).

### Câu 5: Làm thế nào để áp dụng Nguyên lý Khả thế Liskov (Liskov Substitution Principle - chữ L trong SOLID) khi mở rộng DataLoader cho Camera IP RTSP?
* **Gợi ý trả lời của Tech Lead**:
  * Tạo class `RTSPVideoStream` có cùng bộ hàm giao diện public: `start()`, `read()`, `running()`, `stop()`.
  * File `main.py` có thể thay thế `FileVideoStream` bằng `RTSPVideoStream` mà 100% code AI đằng sau không bị ảnh hưởng.

### Câu 6: Làm thế nào một Kỹ sư xác định được những thuộc tính và hàm nào cần có trong class `FileVideoStream` từ con số 0?
* **Gợi ý trả lời của Tech Lead**:
  * Dựa trên **Công thức 3 Cấu phần Vòng đời**:
    1. *Cần gì để tồn tại?* $\to$ Sinh ra Thuộc tính (`self.path`, `self.stream`, `self.Q`, `self.stopped`).
    2. *Bên ngoài cần gì?* $\to$ Sinh ra Hàm Public (`start()`, `read()`, `running()`, `stop()`).
    3. *Nội bộ phải làm gì?* $\to$ Sinh ra Hàm Private (`_update()`).

---

## PHẦN 2: MÔ HÌNH PRODUCER-CONSUMER & THREADING (MULTI-THREADING)

### Câu 7: Mô hình Producer-Consumer trong `FileVideoStream` giải quyết bài toán nghẽn I/O (Disk Bottleneck) như thế nào?
* **Gợi ý trả lời của Tech Lead**:
  * Đọc đĩa (`cv2.read()`) là tác vụ nghẽn I/O.
  * **Producer (Background Thread)**: Chạy ngầm đọc đĩa liên tục nạp 50 frame vào RAM Queue.
  * **Consumer (Vòng lặp AI)**: Chỉ việc bốc ảnh từ RAM Queue ra xử lý với tốc độ tối đa ($> 90$ FPS) mà không bao giờ phải chờ đĩa cứng.

### Câu 8: Tại sao lại đặt thuộc tính `self.thread.daemon = True` khi khởi tạo Thread?
* **Gợi ý trả lời của Tech Lead**:
  * Daemon Thread là thread chạy nền phụ thuộc vào tiến trình chính.
  * Khi tiến trình chính (Main Process) bị tắt hoặc ngắt đột ngột, Daemon Thread sẽ **tự động bị tiêu hủy theo**, tránh hiện tượng Thread "ma" (Zombie Thread) chạy ngầm ngốn CPU.

### Câu 9: Trong hàm `_update()`, tại sao khi Hàng đợi bị đầy (`self.Q.full()`), Thread lại gọi `time.sleep(self.fps_sleep)`?
* **Gợi ý trả lời của Tech Lead**:
  * Nếu không nghỉ `sleep`, vòng lặp `while True` sẽ chạy với tốc độ hàng triệu lần/giây chỉ để kiểm tra `full()`, gây ra hiện tượng **Spinslock ngốn 100% CPU**.
  * `sleep` giúp Thread chủ động nhường nhân CPU cho Consumer bốc ảnh.

### Câu 10: Điểm khác biệt giữa việc chạy `cv2.VideoCapture` trên Thread (trong cùng 1 Process) và chạy trên Process riêng biệt là gì?
* **Gợi ý trả lời của Tech Lead**:
  * Thread chia sẻ chung không gian bộ nhớ RAM với Consumer, giúp truyền mảng Numpy ảnh siêu nhanh không tốn chi phí copy.
  * Tuy nhiên, Thread bị vướng khóa GIL của Python. Do OpenCV release khóa GIL khi decode video, Threading hoạt động cực kỳ hiệu quả cho bài toán đọc đĩa này.

### Câu 11: Chuyện gì xảy ra nếu Bếp Trưởng (Producer) đọc video nhanh hơn rất nhiều so với tốc độ AI (Consumer) tiêu thụ?
* **Gợi ý trả lời của Tech Lead**:
  * Hàng đợi `Queue(maxsize=50)` sẽ bị chạm trần 50 frames.
  * Producer sẽ tự động chuyển sang trạng thái chờ (`sleep`) cho đến khi Consumer bốc bớt ảnh ra khỏi Queue. RAM được bảo vệ tuyệt đối không bị phình to (Buffer Overflow Protection).

### Câu 12: Chuyện gì xảy ra nếu AI (Consumer) chạy nhanh hơn Bếp Trưởng (Producer) đọc đĩa?
* **Gợi ý trả lời của Tech Lead**:
  * Hàng đợi `Queue` sẽ bị rỗng (`qsize == 0`).
  * Hàm `more()` sẽ tự động cho Consumer ngủ ngắn 50ms (thử lại 5 lần) để chờ Producer đọc đĩa nạp ảnh vào.

---

## PHẦN 3: HÀNG ĐỢI ĐỆM RAM QUEUE & AN TOÀN BỘ NHỚ (QUEUE & MEMORY)

### Câu 13: Tại sao lại chọn cấu trúc dữ liệu `queue.Queue` trong thư viện chuẩn Python thay vì danh sách `list` thông thường?
* **Gợi ý trả lời của Tech Lead**:
  * `queue.Queue` là cấu trúc dữ liệu **Thread-safe (An toàn đa luồng)**. Nó tự động bọc các khóa Lock/Condition bên trong.
  * Danh sách `list` trong Python không Thread-safe, nếu 2 luồng cùng `append()` và `pop()` sẽ gây ra rò rỉ bộ nhớ hoặc sập app (Race Condition).

### Câu 14: Tham số `maxsize = 50` của Queue được chọn dựa trên cơ sở nào?
* **Gợi ý trả lời của Tech Lead**:
  * Với video 30 FPS, 50 frames tương đương với bộ nhớ đệm chừng **1.6 giây**.
  * Con số 50 đủ lớn để làm mịn các đợt sụt giảm tốc độ đĩa cứng, nhưng đủ nhỏ để không ngốn quá nhiều dung lượng RAM (50 ảnh 1080p chỉ ngốn chừng ~300MB RAM).

### Câu 15: Tuple dữ liệu được nạp vào Queue gồm những thành phần nào và tại sao lại đóng gói dạng Tuple?
* **Gợi ý trả lời của Tech Lead**:
  * Tuple: `(timestamp, frame_index, frame_matrix)`.
  * Đóng gói Tuple giúp đính kèm mốc thời gian thực tế (`timestamp`) và thứ tự khung hình (`frame_index`) đi liền với bức ảnh, tránh việc thông tin thời gian bị sai lệch khi đi qua các module AI.

### Câu 16: Hiện tượng Race Condition (Tranh chấp tài nguyên) có thể xảy ra ở những thuộc tính nào trong class này và cách phòng tránh?
* **Gợi ý trả lời của Tech Lead**:
  * Biến cờ `self.stopped` được cả Producer (ghi) và Consumer (đọc) truy cập.
  * Trong Python, phép gán boolean `self.stopped = True` là phép tính Nguyên tử (Atomic operation) được bảo vệ bởi GIL, nên an toàn khi đọc/ghi giữa 2 luồng.

### Câu 17: Làm thế nào để đo đạc dung lượng bộ nhớ RAM mà `FileVideoStream` đang chiếm dụng tại một thời điểm?
* **Gợi ý trả lời của Tech Lead**:
  * Lấy số lượng frame hiện có trong Queue (`Q.qsize()`) nhân với dung lượng 1 frame BGR (`height * width * 3` bytes).

### Câu 18: Sự khác biệt giữa cơ chế Queue FIFO (First-In, First-Out) và LIFO (Last-In, First-Out) trong bài toán xử lý video là gì?
* **Gợi ý trả lời của Tech Lead**:
  * `queue.Queue` là **FIFO**: Khung hình nào vào trước sẽ được AI lấy ra xử lý trước, bảo đảm đúng chiều dòng thời gian vật lý.
  * LIFO sẽ làm đảo ngược thời gian (frame mới nhất ra trước), làm hỏng hoàn toàn thuật toán Tracker và ReID.

---

## PHẦN 4: XỬ LÝ FPS, TIMESTAMP & LỌC KHUNG HÌNH (TIME & MATH LOGIC)

### Câu 19: Nêu công thức tính mốc thời gian Timestamp cho từng frame trong hàm `_update()` và giải thích ý nghĩa các biến?
* **Gợi ý trả lời của Tech Lead**:
  * Công thức: $\text{timestamp} = \text{start\_timestamp} + \left(\text{frame\_counter} \times \frac{1}{\text{cam\_fps}}\right)$.
  * `start_timestamp`: Mốc thời gian bắt đầu thực tế của file video (Unix timestamp).
  * `frame_counter`: Số thứ tự frame đã đọc từ đầu file.
  * `1 / cam_fps`: Khoảng thời gian tính bằng giây giữa 2 frame liên tiếp.

### Câu 20: Cơ chế Lọc bỏ bớt khung hình (Frame Skipping) hoạt động như thế nào khi cấu hình `fps < cam_fps`?
* **Gợi ý trả lời của Tech Lead**:
  * Tính khoảng nạp frame: `keep_frame_interval = round(cam_fps / fps)`.
  * Trong vòng lặp đọc đĩa: Nếu `frame_counter % keep_frame_interval != 0` thì gọi `continue` bỏ qua không nạp vào Queue.

### Câu 21: Tại sao phải phân biệt 2 tham số `cam_fps` và `fps` trong hàm `__init__`?
* **Gợi ý trả lời của Tech Lead**:
  * `cam_fps`: Tốc độ quay của camera gốc (ví dụ camera quay 60 FPS).
  * `fps`: Tốc độ mà mô hình AI mong muốn xử lý (ví dụ AI chỉ cần 15 FPS).
  * Việc tách biệt giúp giảm 75% khối lượng tính toán cho GPU mà mốc thời gian Timestamp vẫn chuẩn xác $100\%$.

### Câu 22: Điều gì xảy ra với Timestamp nếu file video bị mất khung hình (Dropped Frames) từ lúc quay trên đĩa cứng?
* **Gợi ý trả lời của Tech Lead**:
  * Công thức tính dựa trên `frame_counter` giả định thời gian giữa các frame là đều nhau.
  * Nếu video bị mất frame trên đĩa, mốc thời gian có thể bị lệch nhẹ vài millisecond. Trong thực tế cao cấp, người ta đọc trực tiếp mốc timestamp tích hợp trong container mkv/mp4 via OpenCV `cv2.CAP_PROP_POS_MSEC`.

### Câu 23: Tại sao lại dùng `time.time()` làm mốc thời gian mặc định khi `current_time` không được truyền vào?
* **Gợi ý trả lời của Tech Lead**:
  * Khi không có mốc thời gian lịch sử, hệ thống lấy mốc thời gian hiện tại của đồng hồ máy tính (System Epoch Time) để làm gốc tọa độ thời gian cho video.

### Câu 24: Tại sao trong hàm `_update()` người ta lại đo `self.total_capturing_time`?
* **Gợi ý trả lời của Tech Lead**:
  * Dùng để đo đạc hiệu năng (Profiling): Tổng thời gian đĩa cứng tốn để đọc toàn bộ video, giúp Kỹ sư biết đĩa cứng có đang bị nghẽn (I/O Bottleneck) hay không.

---

## PHẦN 5: QUẢN LÝ VÒNG ĐỜI & XỬ LÝ LỖI (LIFECYCLE & TROUBLESHOOTING)

### Câu 25: Nêu chi tiết lý do tại sao hàm `running()` lại kết hợp 2 điều kiện bằng phép `OR`: `return self.more() or not self.stopped`?
* **Gợi ý trả lời của Tech Lead**:
  * **Nếu chỉ dùng `not self.stopped`**: Khi Producer vừa đọc xong frame cuối (stopped = True), vòng while sẽ thoát ngay, làm **mất trắng 10-20 frame cuối** còn nằm trong Queue.
  * **Nếu chỉ dùng `self.more()`**: Ở millisecond đầu tiên khi vừa start, Queue chưa kịp nạp frame nào (`more() == False`), vòng while sẽ **tự ngắt app ngay khi vừa khởi động**.
  * **Phép `OR`**: Đảm bảo app **không sập lúc khởi động** và **không mất frame lúc kết thúc**.

### Câu 26: Hàm `more()` xử lý trường hợp Queue bị trống tạm thời như thế nào để Consumer không bị thoát sớm?
* **Gợi ý trả lời của Tech Lead**:
  * Hàm có một vòng lặp kiên nhẫn: `while self.Q.qsize() == 0 and not self.stopped and tries < 5: time.sleep(0.05)`.
  * Cho Consumer thử lại 5 lần (mỗi lần nghỉ 50ms) để chờ Producer đọc đĩa nạp ảnh vào trước khi kết luận là hết video.

### Câu 27: Tiến trình dọn dẹp tài nguyên trong hàm `stop()` diễn ra theo thứ tự như thế nào?
* **Gợi ý trả lời của Tech Lead**:
  1. Đặt cờ `self.stopped = True` để tín hiệu dừng đến Producer Thread.
  2. Gọi `self.thread.join()` để chờ Producer Thread kết thúc hoàn toàn.
  3. Producer Thread tự động gọi `self.stream.release()` đóng file video OpenCV an toàn.

### Câu 28: Điều gì xảy ra nếu file video bị hỏng giữa chừng (Corrupted File) khi Producer đang đọc?
* **Gợi ý trả lời của Tech Lead**:
  * OpenCV `stream.read()` sẽ trả về `frame = None` hoặc `grabbed = False`.
  * Đoạn code kiểm tra `if frame is None: self.stopped = True; break` sẽ kích hoạt, dừng luồng đọc ngầm an toàn và thông báo hết video mà không gây ra crash app.

### Câu 29: Kịch bản kiểm thử (Test Scenario) nào là quan trọng nhất để đảm bảo `FileVideoStream` hoạt động 100% không có bug?
* **Gợi ý trả lời của Tech Lead**:
  * **Test 1**: Đọc hết file video ngắn (100 frames) và đếm xem Consumer có bốc ra đủ đúng 100 frames hay không.
  * **Test 2**: Thử gọi `stop()` đột ngột giữa chừng xem Thread có bị treo hoang (Zombie Thread) hay không.
  * **Test 3**: Giả lập Consumer chạy cực chậm (`time.sleep(0.1)`) xem RAM Queue có bị phình to quá 50 frames hay không.

### Câu 30: Bạn đánh giá một module Video DataLoader đã đạt chuẩn Production-Ready dựa trên những chỉ số nào?
* **Gợi ý trả lời của Tech Lead**:
  * **Chỉ số 1 (Zero Frame Loss)**: Đọc không sót bất kỳ khung hình nào từ file đĩa.
  * **Chỉ số 2 (Throughput)**: Tốc độ nạp frame vào Queue đạt $\ge 90$ FPS.
  * **Chỉ số 3 (Resource Leak)**: RAM giữ nguyên ở mức cố định (~300MB), CPU tiêu thụ $< 5\%$ cho việc đọc đĩa, 0% rò rỉ Thread khi dừng.

---
*Bộ 30 câu hỏi phỏng vấn và tự kiểm tra chuyên sâu module Video DataLoader được soạn bởi Senior Tech Lead.*
