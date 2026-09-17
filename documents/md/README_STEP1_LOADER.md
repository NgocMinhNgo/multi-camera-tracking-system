# BƯỚC 1: FILE VIDEO STREAM LOADER (PRODUCER-CONSUMER PATTERN)

> **Thư mục code mới**: `my_reid_project/`  
> **Mục tiêu**: Xây dựng luồng đọc video đĩa bất đồng bộ sử dụng Background Thread & Thread-safe Queue.

---

## 1. Tư duy First Principles của Bước 1

```text
[ File Video (.mp4 / .mkv) ]
            │
            ▼
[ Producer Thread (FileVideoStream._update) ]  <--- Đọc liên tục từ đĩa cứng
            │
            ▼ Nạp tuple (timestamp, frame_index, frame_matrix)
    [ Queue(maxsize=50) ]                      <--- Bộ nhớ đệm RAM Thread-safe
            │
            ▼ Lấy từng frame ra xử lý (fvs.read())
[ Consumer (Vòng lặp AI Process) ]            <--- Chạy với tốc độ tối đa không bị nghẽn đĩa!
```

### Tại sao không đọc trực tiếp `cv2.VideoCapture.read()` trong vòng lặp AI chính?
1. **Nghẽn đĩa I/O (Disk Bottleneck)**: Phép đọc khung hình từ đĩa cứng là phép tính nghẽn I/O. Nếu vòng lặp AI vừa đọc đĩa vừa chạy YOLOv5, GPU sẽ bị bỏ trống trong lúc chờ đĩa cứng trả về bức ảnh.
2. **Giải pháp Producer-Consumer**: 
   * **Producer (Background Thread)**: Chạy ngầm 24/7 chỉ để đọc sẵn 50 khung hình nạp vào đĩa RAM (`Queue`).
   * **Consumer (Vòng lặp AI)**: Chỉ việc bốc ảnh từ RAM ra xử lý với tốc độ tối đa ($> 90$ FPS)!

---

## 2. Giải thích Chi tiết Mã nguồn `FileVideoStream`

File code chính: [`my_reid_project/modules/loader/file_stream.py`](file:///c:/Users/Admin/Downloads/tmp_prj/my_reid_project/modules/loader/file_stream.py)

### Các thuộc tính cốt lõi:
* `self.stream`: Đối tượng OpenCV `cv2.VideoCapture(path, cv2.CAP_FFMPEG)`.
* `self.Q`: Hàng đợi `queue.Queue(maxsize=50)` thread-safe.
* `self.thread`: Thread chạy ngầm hàm `_update()`.
* `self.start_timestamp`: Mốc thời gian thực tế bắt đầu của video.

### Công thức tính Timestamp chuẩn thời gian thực cho từng frame:
$$\text{timestamp} = \text{start\_timestamp} + \text{frame\_count} \times \frac{1}{\text{cam\_fps}}$$

---

## 3. Hướng dẫn Chạy thử Kiểm thử

Bạn chỉ cần chạy lệnh sau trên terminal:

```bash
python my_reid_project/test_stream_loader.py
```

### Kết quả chạy kiểm thử thực tế:
* Script tự động tạo file video giả lập `synthetic_test_video.mp4` (150 frames) chứa ô vuông di chuyển.
* Producer Thread đọc video và nạp vào Queue.
* Consumer bốc frame ra xử lý đạt tốc độ **$94.4$ FPS** mượt mà!
