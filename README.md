# SystemTrafficLaw
# Traffic Violation Detection System using Deep Learning

## Giới thiệu

Dự án này xây dựng hệ thống phát hiện hành vi vi phạm giao thông dựa trên Computer Vision, Deep Learning và Tracking. 

Hệ thống có khả năng:
- Phát hiện và theo dõi phương tiện giao thông
- Phát hiện hành vi vượt đèn đỏ
- Phát hiện không đội mũ bảo hiểm và chở quá số người (segmentation người / đầu / mũ)
- Tự động chụp ảnh phương tiện vi phạm

Hệ thống được thiết kế theo **pipeline hai mô hình** (vehicle detection + hybrid segmentation), phản ánh quy trình camera giao thông thông minh trong thực tế.

## Mục tiêu dự án

- Ứng dụng YOLO + Tracking + Segmentation hybrid (CNN + attention) vào bài toán giao thông
- Hai mô hình chuyên biệt: phương tiện và người–mũ–đầu
- Xây dựng pipeline từ video → phát hiện vi phạm → bằng chứng hình ảnh
- Phục vụ mục đích nghiên cứu – học tập – demo hệ thống giám sát giao thông

## Kiến trúc tổng thể hệ thống

```
Camera / Video
      ↓
Model 1: Vehicle Detection + Tracking
      ↓
Phát hiện hành vi vi phạm (logic) + Trigger Capture (khi vi phạm)
      ↓
Model 2: YOLOv8-Seg + CBAM + ViTs (helmet / head / person)
      ↓
Lưu DB & Xuất báo cáo vi phạm
```

## Các mô hình trong hệ thống

### Model 1 – Vehicle Detection & Tracking

**Nhiệm vụ**
- Phát hiện phương tiện và người tham gia giao thông
- Theo dõi đối tượng qua nhiều frame
- Phục vụ phát hiện hành vi vượt đèn đỏ

**Công nghệ**
- YOLOv8 (Detection hoặc Segmentation)
- DeepSORT / ByteTrack

**Class label (checkpoint `vietnam_vehicle_v2`)**
- `car`, `motocycle` (ghi chú: tên lớp trong dataset), `truck`, `bus`

**Output**
- Bounding box / mask
- Track ID
- Quỹ đạo di chuyển

#### Kết quả huấn luyện – `VehicleModel/vietnam_vehicle_v2`

Dữ liệu huấn luyện phản ánh giao thông Việt Nam: lớp **xe máy** và **ô tô** chiếm đa số; **xe tải** và **xe buýt** ít mẫu hơn (mất cân bằng lớp). Phần lớn bbox có kích thước nhỏ trên khung hình (xe ở xa / góc rộng).

![Phân bố lớp và thống kê bbox – labels](VehicleModel/vietnam_vehicle_v2/labels.jpg)

**Đường cong Precision–Recall (mAP@0.5 theo lớp)**

| Lớp | mAP@0.5 |
|-----|---------|
| car | 0.946 |
| motocycle | 0.905 |
| truck | 0.904 |
| bus | 0.795 |
| **Trung bình (all classes)** | **0.888** |

![Precision–Recall curve](VehicleModel/vietnam_vehicle_v2/BoxPR_curve.png)

**Đường cong F1 theo ngưỡng confidence**

- Điểm tối ưu gợi ý (trung bình mọi lớp): **F1 ≈ 0.84** tại **confidence ≈ 0.394** (điều chỉnh theo ưu tiên precision hay recall khi triển khai).

![F1–Confidence curve](VehicleModel/vietnam_vehicle_v2/BoxF1_curve.png)

**Precision và Recall theo confidence**

![Precision–Confidence curve](VehicleModel/vietnam_vehicle_v2/BoxP_curve.png)

![Recall–Confidence curve](VehicleModel/vietnam_vehicle_v2/BoxR_curve.png)

**Ma trận nhầm lẫn**

- Lớp **bus** yếu nhất so với các lớn còn lại; có nhầm **bus → car** và tỉ lệ **bus → background**.
- Cần chú ý **false positive**: vùng nền đôi khi bị dự đoán thành `car` / `motocycle` — có thể bổ sung mẫu nền âm (hard negatives) hoặc tinh chỉnh ngưỡng confidence.

![Confusion matrix (normalized)](VehicleModel/vietnam_vehicle_v2/confusion_matrix_normalized.png)

![Confusion matrix (số đếm)](VehicleModel/vietnam_vehicle_v2/confusion_matrix.png)

### Model 2 – Segmentation người / đầu / mũ (Hybrid YOLOv8-Seg + CBAM + ViTs)

**Nhiệm vụ**
- Phân đoạn (mask) **person**, **head**, **helmet** để suy ra không đội mũ và ước lượng số người trên xe
- Phát hiện hành vi chở quá số người (kết hợp logic với Model 1)

**Kiến trúc: mô hình hybrid hoàn chỉnh**

| Thành phần | Vai trò |
|------------|--------|
| **ViTs / Transformer (Backbone)** | Thu ngữ cảnh toàn cục, hỗ trợ vật thể nhỏ / khó (che khuất, nền rối) |
| **CBAM (Neck / Head)** | Tinh chỉnh không gian–kênh, giảm nhiễu, cải thiện độ sắc nét biên mặt nạ |

**Cộng hưởng (synergy):** Transformer bù đắp giới hạn receptive field của CNN thuần; CBAM sau đó làm nổi bật vùng có ý nghĩa và ổn định mask so với chỉ dùng CNN.

**Công nghệ**
- YOLOv8-Seg làm khung detection/segmentation
- CBAM nhúng ở neck/head theo thiết kế của bạn
- Khối ViT/Swin (hoặc tương đương) ở backbone

**Dữ liệu đầu vào**
- Ảnh / crop từ vùng quan tâm sau Model 1 (ví dụ xe máy + người)

**Class label**
- `person`
- `head`
- `helmet`

**Logic vi phạm (gợi ý)**

*Không đội mũ:* ≥ N frame liên tiếp không có `helmet` trên người cướp xe → vi phạm (N tùy cấu hình).

*Chở quá số người:* một phương tiện (ví dụ xe máy từ Model 1) + số instance `person` vượt ngưỡng quy định → vi phạm.

## Xử lý & tổ chức dữ liệu

### 1. Dữ liệu thô
- Video giao thông từ camera
- Trích xuất frame theo FPS phù hợp

### 2. Tiền xử lý
- Loại bỏ ảnh quá mờ (Laplacian variance)
- Resize ảnh về kích thước chuẩn
- Augmentation: flip, blur, noise

### 3. Annotation

**Lưu ý:** Các model dùng chung dữ liệu ảnh/video, **KHÔNG** dùng chung label

```
raw_images/
labels_model1/
labels_model2/
labels_model3/
```

## Chiến lược huấn luyện (Training Strategy)

### Fine-tuning
- Sử dụng pretrained YOLOv8
- Freeze backbone giai đoạn đầu
- Fine-tune head theo từng bài toán
- Batch size & learning rate điều chỉnh theo GPU

### Loss function
- YOLO default loss (box + cls + dfl)
- Mask loss (nếu dùng segmentation)

## Đánh giá mô hình (Evaluation Metrics)

### Detection / Segmentation
- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
- IoU

### Tracking
- MOTA
- ID Switch
- FPS

### OCR
- Character Accuracy
- Plate-level Accuracy

## Cơ chế phát hiện & ghi nhận vi phạm

- Hệ thống chỉ chụp ảnh khi có vi phạm
- Mỗi lỗi là module độc lập
- Một phương tiện có thể vi phạm nhiều lỗi cùng lúc

**Ví dụ:**
```
Không vượt đèn đỏ ❌
Nhưng:
Không đội mũ bảo hiểm ✅
→ Vẫn ghi nhận vi phạm
```

## Kết quả đầu ra

- Ảnh phương tiện vi phạm
- Biển số đã nhận dạng
- Thời gian & loại vi phạm
- Dữ liệu sẵn sàng hiển thị dashboard hoặc báo cáo

## Hướng phát triển

- Nhận diện đi ngược chiều
- Nhận diện đi sai làn
- Tối ưu realtime (TensorRT)
- Kết nối hệ thống IoT / Smart City

## Công nghệ sử dụng

- Python
- YOLOv8
- OpenCV
- DeepSORT / ByteTrack
- PaddleOCR / EasyOCR
- ESRGAN / Real-ESRGAN
