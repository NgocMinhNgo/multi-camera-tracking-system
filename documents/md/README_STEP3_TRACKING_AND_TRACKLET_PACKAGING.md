# BƯỚC 3: HƯỚNG DẪN LẬP TRÌNH SINGLE-CAM TRACKING, BỘ NHỚ RAM ĐỆM VÀ ĐÓNG GÓI TRACKLET LƯU MONGODB

> **Tài liệu hướng dẫn phát triển**: Step-by-Step Single-Cam Tracking & Tracklet Packaging Guide  
> **Áp dụng cho thư mục**: `multi-camera-tracking-system/`  
> **Dựa trên chuẩn thiết kế**: [`OFFLINE_MULTICAM_TRACKING_GUIDE.md`](file:///D:/tmp_prj/documents/OFFLINE_MULTICAM_TRACKING_GUIDE.md#L92-L117) và codebase tham chiếu [`src/modules/track_manager/person_track_manager.py`](file:///d:/tmp_prj/tmp_prj/src/modules/track_manager/person_track_manager.py)

---

## 1. TỔNG QUAN KIẾN TRÚC XỬ LÝ SINGLE-CAM TRACKLET

Hệ thống xử lý dấu vết camera đơn được chia làm **2 Giai đoạn độc lập**:
1. **Giai đoạn 1 (Frame-Level Processing - Liên tục 30 FPS)**: Đọc từng frame, gọi SORT Tracker cấp ID tạm, lọc ảnh crop nét và lưu đệm vào CSDL RAM (`dict_tracks`).
2. **Giai đoạn 2 (Tracklet-Level Processing - Chạy khi ID Chết)**: Khi khách hàng bước ra khỏi camera, gom toàn bộ ảnh crop trong RAM trích xuất Vector FastReID + View Classifier, gom nhóm Clustering chọn Top vector đẹp nhất, đóng gói mã hóa lưu MongoDB và giải phóng RAM.

```text
====================================================================================
GIAI ĐOẠN 1: VÒNG LẶP TỪNG FRAME (Chạy liên tục 30 FPS)
====================================================================================
[ Frame t ] ──► [ Loader ] ──► [ YOLO Detector ] ──► [ SORT Tracker ]
                                                           │
                                                           ▼
                                            Cập nhật BBox & cấp ID tạm: `Track #5`
                                                           │
                                                           ▼
                                            Cắt ảnh người (Crop Image)
                                                           │
                                                           ▼
                                            [ Selection (is_good_box) ]
                                            Lưu ảnh crop đạt chuẩn vào RAM `dict_tracks[5]`

====================================================================================
GIAI ĐOẠN 2: KHI NGƯỜI BƯỚC RA KHỎI CAM (Chạy 1 lần khi ID Chết hẳn)
====================================================================================
                                            `Track #5` Chết hẳn (Khi dead_time > max_fragment_time)
                                                           │
                                                           ▼
                                            [ FastReID Vectorizer & View Classifier ]
                                            Trích mảng Vector 256-dim + Nhãn ('front'/'side'/'back')
                                                           │
                                                           ▼
                                            [ Clustering (MiniBatchKMeans / DBSCAN) ]
                                            Lọc ra Top 5-10 Vector đại diện xuất sắc nhất
                                                           │
                                                           ▼
                                            [ Mã hóa Base64 & Lưu MongoDB `TrackResult` ]
                                            Lưu vào DB ➔ Xóa `Track #5` khỏi RAM để giải phóng bộ nhớ!
```

---

## 2. GIAI ĐOẠN 1: HƯỚNG DẪN XÂY DỰNG VÒNG LẶP TỪNG FRAME

Giai đoạn này diễn ra liên tục trên từng khung hình ($30\text{ FPS}$).

### 🔹 Bước 1.1: Tích hợp SORT Tracker (`PersonTracker.track`)

* **Vị trí mã nguồn**: [`src/modules/tracking/person_tracking.py`](file:///d:/tmp_prj/tmp_prj/src/modules/tracking/person_tracking.py) & [`inference_engine.py:L198`](file:///d:/tmp_prj/tmp_prj/src/processes/inference_engine.py#L198)
* **Đầu vào (Input)**: 
  - `frame`: Khung hình OpenCV Numpy `[H, W, 3]`.
  - `person_results`: Danh sách các đối tượng `DetectionResult` từ YOLOv5/v8 (`box=[x1,y1,x2,y2]`, `score`, `classname='person'`).
* **Trình tự code thực thi**:
  ```python
  person_alive_tracks, person_dead_tracks = self.person_tracker.track(
      person_results, self.person_track_manager, timestamp, frame
  )
  ```
* **Thuật toán bên trong SORT Tracker**:
  1. **Kalman Filter**: Dự đoán vị trí BBox của các ID cũ ở khung hình $t$ dựa trên vận tốc di chuyển.
  2. **Hungarian Algorithm**: Ghép nối IoU (Intersection over Union) giữa BBox dự đoán và BBox thực tế từ YOLO.
* **Đầu ra (Output)**: 
  - `person_alive_tracks`: Danh sách các đối tượng `TrackData` đại diện cho các ID còn đang xuất hiện trong camera.
  - Cấu trúc mỗi `TrackData` gồm:
    - `track_id`: ID tạm thời (ví dụ: `5`).
    - `box`: Tọa độ `[x1, y1, x2, y2]`.
    - `cropped_img`: Bức ảnh người cắt từ frame gốc: `frame[y1:y2, x1:x2]`.

---

### 🔹 Bước 1.2: Quản lý Bộ nhớ RAM Đệm & Lọc Ảnh Crop (`update_session_tracks`)

* **Vị trí mã nguồn**: [`src/modules/track_manager/person_track_manager.py:L76`](file:///d:/tmp_prj/tmp_prj/src/modules/track_manager/person_track_manager.py#L76)
* **Đầu vào (Input)**: `person_alive_tracks`, `timestamp`, `frame`.
* **Trình tự code thực thi**:
  ```python
  for track_data in alive_tracks:
      track_id = track_data.track_id
      
      # 1. Tạo mới hoặc cập nhật thông tin Track trong RAM dict_tracks
      if track_id in self.dict_tracks:
          self.dict_tracks[track_id].end_time = timestamp
      else:
          new_track = PersonTrackInfo(track_id=track_id, timestamp=timestamp, ...)
          self.dict_tracks[track_id] = new_track
      
      # 2. Kiểm tra & Lưu tạm ảnh Crop đạt tiêu chuẩn vào RAM
      self.update_cropped_imgs(frame, track_data, track_id)
  ```
* **Chi tiết hàm `update_cropped_imgs` (Selection ở cấp độ Frame)**:
  ```python
  def update_cropped_imgs(self, frame, track_data, track_id):
      center_point = self.get_center_point(frame)
      # Kiểm tra box có đủ lớn (> 15px) và nằm ở vùng nét trung tâm không
      good_box = track_data.is_good_box(center_point[0], self.config['SELECTION'])
      
      if good_box and min(track_data.cropped_img.shape[0], track_data.cropped_img.shape[1]) > 15:
          # LƯU TẠM ẢNH CROP VÀO MẢNG TRONG RAM (Chưa chạy FastReID!)
          self.dict_tracks[track_id].cropped_imgs.append(track_data.cropped_img)
  ```
* **Đầu ra (Output)**: Một danh sách mảng ảnh crop `cropped_imgs` được tích lũy liên tục vào bộ nhớ RAM của `Track #5`.

---

## 3. GIAI ĐOẠN 2: HƯỚNG DẪN XÂY DỰNG XỬ LÝ KHI TRACK CHẾT (DEAD TRACKLET PACKAGING)

Giai đoạn này **chỉ kích hoạt 1 lần duy nhất** khi người đó bước ra khỏi góc quay camera.

### 🔹 Bước 2.1: Phát hiện ID Đã Chết Hẳn (`get_last_dead_tracks`)

* **Vị trí mã nguồn**: [`inference_engine.py:L250`](file:///d:/tmp_prj/tmp_prj/src/processes/inference_engine.py#L250) & [`person_track_manager.py:L195`](file:///d:/tmp_prj/tmp_prj/src/modules/track_manager/person_track_manager.py#L195)
* **Đầu vào (Input)**: `timestamp` của frame hiện tại.
* **Trình tự code thực thi**:
  ```python
  # Kiểm tra xem có Track nào mất dấu quá max_fragment_time (VD: > 3 giây không xuất hiện lại)
  real_dead_tracks = self.person_track_manager.get_last_dead_tracks(timestamp)
  ```
* **Công thức xác định**: `timestamp - track_info.dead_time > max_fragment_time`
* **Đầu ra (Output)**: Danh sách `real_dead_tracks` chứa các đối tượng `PersonTrackInfo` đã thực sự kết thúc hành trình (ví dụ: `Track #5`).

---

### 🔹 Bước 2.2: Trích Vector FastReID & Phân loại Góc nhìn (`update_emb_of_track`)

* **Vị trí mã nguồn**: [`person_track_manager.py:L116`](file:///d:/tmp_prj/tmp_prj/src/modules/track_manager/person_track_manager.py#L116)
* **Đầu vào (Input)**: `track_id = 5` (Lấy ra danh sách `cropped_imgs` chứa hàng trăm ảnh crop tích lũy trong RAM).
* **Trình tự code thực thi**:
  ```python
  def update_emb_of_track(self, track_id):
      if not self.dict_tracks[track_id].cropped_imgs:
          return
          
      # 1. Gọi Triton Model FastReID Vectorizer & View Classifier
      embs, person_views, up_bodys = self.vectorizer.get_infer_result(
          self.dict_tracks[track_id].cropped_imgs
      )
      
      # 2. Chạy Clustering (MiniBatchKMeans/DBSCAN) gom nhóm lọc lấy các Vector đại diện tốt nhất
      clusters = self.cluster.fit(self.dict_tracks[track_id].embeddings)
      
      # (Giữ lại Top 5-10 Vector đại diện tốt nhất, loại bỏ vector rác/nhiễu)
      self.dict_tracks[track_id].embeddings = selected_embs
      self.dict_tracks[track_id].person_views = selected_person_views
      
      # 3. Dọn dẹp mảng ảnh crop trong RAM để giải phóng bộ nhớ
      self.dict_tracks[track_id].cropped_imgs = []
  ```
* **Đầu ra (Output)**: 
  - `embeddings`: Mảng các Vector đặc trưng 256 chiều `[Top 5-10 vectors]`.
  - `person_views`: Danh sách nhãn góc nhìn tương ứng `['front', 'side', 'back', ...]`.

---

### 🔹 Bước 2.3: Mã Hóa & Lưu MongoDB `TrackResult`

* **Vị trí mã nguồn**: [`inference_engine.py:L260-L264`](file:///d:/tmp_prj/tmp_prj/src/processes/inference_engine.py#L260-L264)
* **Đầu vào (Input)**: Đối tượng `trk` (`Track #5` đã hoàn tất trích xuất Top 5-10 Vector 256-dim).
* **Trình tự code thực thi**:
  ```python
  # 1. Mã hóa dữ liệu Tracklet (vị trí, thời gian, vectors) thành chuỗi base64
  encoded_data = encode_data(trk.__repr__())

  # 2. Tạo bản ghi MongoEngine và lưu vào CSDL MongoDB
  new_track_info = TrackResult(track_info=encoded_data)
  new_track_info.save()

  # 3. XÓA HOÀN TOÀN TRACK #5 KHỎI RAM ĐỂ GIẢI PHÓNG BỘ NHỚ
  self.person_track_manager.remove_track(trk.track_id)
  ```
* **Đầu ra (Output)**: 
  - 1 bản ghi mới lưu vào MongoDB collection `TrackResult`.
  - Bộ nhớ RAM của Python cho `Track #5` được giải phóng hoàn toàn $100\%$!

---

## 4. BẢNG TÓM TẮT CHECKLIST LẬP TRÌNH VÀ BẢO TRÌ BỘ NHỚ

| Tiêu chí | Giai đoạn 1: Trong vòng lặp Frame | Giai đoạn 2: Khi Track chết hẳn |
| :--- | :--- | :--- |
| **Tần suất thực thi** | 30 lần / giây (Mỗi frame) | 1 lần duy nhất cho mỗi ID khách hàng |
| **Nơi lưu dữ liệu** | Bộ nhớ RAM tạm thời (`self.dict_tracks`) | Cơ sở dữ liệu đĩa MongoDB (`TrackResult`) |
| **Phép tính chính** | YOLO Detect BBox & SORT Kalman Match | FastReID Tensor (256-dim) & Clustering |
| **Bảo vệ RAM** | Chỉ `append` các ảnh crop đạt `is_good_box` | Gọi `remove_track()` giải phóng RAM ngay sau khi `.save()` |
| **Mục đích cốt lõi** | Giữ vết di chuyển & gom tập ảnh crop | Nén dữ liệu tối giản sẵn sàng cho ReID đa camera |

---
*Tài liệu Hướng dẫn Lập trình Single-Cam Tracking & Tracklet Packaging thuộc Bước 3 trong Chuỗi phát triển Hệ thống Multi-Camera Tracking.*
