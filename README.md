# SystemTrafficLaw

## Hệ thống phát hiện vi phạm giao thông dựa trên học sâu và pipeline đa mô hình

**Traffic Violation Detection System using Deep Learning and Multi-Model Pipeline**

---

## Tóm tắt (Abstract)

Nghiên cứu hướng tới **kiến trúc YOLOv8-Seg hybrid** (ViT ở backbone, CBAM ở neck/head) cho phân đoạn **người / đầu / mũ** — đây là **trọng tâm đề tài** và **đang trong quá trình triển khai** (huấn luyện, tinh chỉnh YAML, tích hợp Ultralytics), **chưa hoàn tất** tại thời điểm báo cáo.

Song song, pipeline sử dụng một **mô-đun phát hiện phương tiện** (YOLOv8 + tracker) chỉ với vai trò **phụ trợ**: định vị ROI, theo dõi, logic vượt đèn — **không phải đối tượng nghiên cứu chính**. Các số liệu mAP và hình ảnh đánh giá trên checkpoint `vietnam_vehicle_v2` được đính kèm như **tài liệu tham khảo / baseline phụ**, không thay thế cho kết quả của mô hình hybrid.

Hướng mở rộng: hoàn thiện thực nghiệm hybrid, nhận dạng biển số, tối ưu suy luận thời gian thực.

**Từ khóa (Keywords):** phát hiện vi phạm giao thông; YOLOv8; phân đoạn thể hiện; CBAM; Vision Transformer; theo dõi đa đối tượng.

---

## 1. Giới thiệu (Introduction)

Giao thông đô thị tại Việt Nam đặc trưng bởi mật độ xe máy cao, cảnh quan phức tạp và nhiễu nền, khiến các hệ thống chỉ dựa trên quy tắc thủ công khó mở rộng quy mô. Các phương pháp học sâu cho phép học trực tiếp từ dữ liệu ảnh/video; riêng bài toán **mũ–đầu–người**, phân đoạn mức pixel phù hợp hơn so với chỉ hộp giới hạn khi cần suy luận chi tiết vùng đội mũ và số người trên xe.

**Đóng góp dự kiến / trọng tâm:**

1. **Thiết kế và hiện thực hóa** kiến trúc **YOLOv8-Seg + ViT + CBAM** cho ba lớp ngữ nghĩa — phần này **đang làm dở**, sẽ bổ sung kết quả định lượng khi hoàn tất.
2. **Pipeline hai tầng** trong đó tầng phương tiện chỉ là **tiền xử lý phụ trợ**; tầng hybrid mới mang tính **đề xuất phương pháp** của luận văn / báo cáo.
3. **Mô-đun vehicle** (`vietnam_vehicle_v2`): có số liệu tham khảo — **không** coi là đóng góp lý thuyết chính, chỉ phục vụ minh họa pipeline và đối chiếu triển khai.

---

## 2. Phương pháp đề xuất (Proposed Method)

### 2.1. Pipeline tổng thể

Luồng xử lý từ camera đến lưu trữ vi phạm được mô tả như sau:

```
Camera / Video
      ↓
[Phụ trợ] Vehicle Detection + Tracking
      ↓
Phát hiện hành vi vi phạm (logic) + Trigger Capture (khi vi phạm)
      ↓
[Trọng tâm — đang triển khai] YOLOv8-Seg + CBAM + ViT — helmet / head / person
      ↓
Lưu DB & Xuất báo cáo vi phạm
```

**Tầng phương tiện** chỉ cung cấp bbox/track ID phục vụ ROI và luật giao thông. **Tầng hybrid** là nơi xử lý chính cho mũ–đầu–người; trạng thái hiện tại: **chưa xong** (xem §3, §5).

### 2.2. Động lực hybrid CNN–Transformer–Attention

- **CNN (YOLO backbone/neck):** hiệu quả tính toán, đặc trưng đa tỉ lệ phù hợp phát hiện thời gian thực.
- **ViT ở backbone (nhánh độ phân giải thấp, ví dụ P5):** mô hình hóa phụ thuộc xa, hỗ trợ vật thể nhỏ hoặc bị che khuất trong ngữ cảnh đông đúc.
- **CBAM ở nhiều mức (P3, P4, P5 và neck):** làm nổi bật kênh và vị trí không gian, giảm nhiễu nền, cải thiện biên mặt nạ so với backbone thuần CNN.

---

## 3. Kiến trúc hybrid YOLOv8–CBAM–ViT (trọng tâm đề tài — *work in progress*)

**Trạng thái:** đang xây dựng / nhúng module (YAML tùy chỉnh, chỉnh `parse_model`, huấn luyện, đánh giá). **Chưa có** bộ số liệu đầy đủ tương đương mục phương tiện bên dưới.

Cấu hình tham chiếu (mục tiêu): `yolov8_hybrid_cbam_vit.yaml`. Đầu vào: tensor ảnh (3 kênh, H×W). Đầu ra mong muốn: **bản đồ phân đoạn ba lớp** `person`, `head`, `helmet`.

![Hình 1. Kiến trúc YOLOv8 Hybrid CBAM–ViT Segmentation (3 lớp)](image/Gemini_Generated_Image_d2hbaqd2hbaqd2hb%20%281%29.png)

**Bảng 1. Tóm tắt khối chức năng**

| Thành phần | Mô tả |
|------------|--------|
| Backbone | Chuỗi Conv/C2f; đặc trưng đa tỉ lệ P3 (stride 8), P4 (stride 16), P5 (stride 32). CBAM sau P3, P4; tại P5: ViT (1024 kênh) → SPPF → CBAM. |
| Neck | Upsample, Concat, C2f (chuẩn hóa kênh giữa backbone và head). |
| Head | Module Segment (tham số dạng `[nc: 3, 32, 3]`), sinh mặt nạ phân đoạn. |

**Chú thích màu trên sơ đồ:** Conv (xanh dương), C2f (xanh lá), SPPF (cam), CBAM (vàng), Segment (đỏ đậm).

---

## 4. Module phụ trợ: phát hiện phương tiện và theo dõi *(không phải trọng tâm nghiên cứu)*

Mục này mô tả **YOLOv8 detection chuẩn** trên tập `vietnam_vehicle_v2` nhằm **hỗ trợ pipeline** (ROI, tracking, vượt đèn). Đây **không** phải đóng góp phương pháp chính của đề tài; các biểu đồ và bảng sau chỉ mang tính **tham khảo / baseline triển khai**.

### 4.1. Bài toán và nhãn

- **Nhiệm vụ:** phát hiện lớp phương tiện, theo dõi khung thời gian, hỗ trợ logic vượt đèn đỏ.
- **Công nghệ:** YOLOv8 (detection); DeepSORT / ByteTrack.
- **Lớp (checkpoint `vietnam_vehicle_v2`):** `car`, `motocycle`, `truck`, `bus` (ghi chú: tên lớp `motocycle` theo dataset).
- **Đầu ra:** bounding box, track ID, quỹ đạo (tùy cấu hình tracker).

### 4.2. Đặc điểm dữ liệu huấn luyện

Tập huấn luyện phản ánh giao thông Việt Nam: **xe máy** và **ô tô** chiếm đa số; **xe tải** và **xe buýt** thưa hơn (mất cân bằng lớp). Phân bố bbox thiên về **đối tượng nhỏ trên khung hình** (xe ở xa, góc camera rộng).

![Hình 2. Thống kê phân bố lớp và bbox — labels](VehicleModel/vietnam_vehicle_v2/labels.jpg)

![Hình 3. Batch huấn luyện (mosaic); nhãn 0–3: car, motocycle, truck, bus](VehicleModel/vietnam_vehicle_v2/train_batch0.jpg)

### 4.3. Quá trình huấn luyện (giai đoạn ~150 epoch)

Loss huấn luyện giảm; trên tập kiểm định, `metrics/mAP50(B)` khoảng **0.86–0.87**, `mAP50-95(B)` khoảng **0.68** ở cuối quá trình. **Thảo luận:** nếu `val/cls_loss` **tăng** trong khi `train/cls_loss` giảm, có dấu hiệu **quá khớp (overfitting)** nhánh phân lớp; có thể xem xét tăng tăng cường dữ liệu, dừng sớm hoặc điều chỉnh regularization.

![Hình 4. Loss và metrics theo epoch](VehicleModel/vietnam_vehicle_v2/results.png)

### 4.4. Kết quả định lượng

**Bảng 2. mAP@0.5 theo lớp (validation)**

| Lớp | mAP@0.5 |
|-----|---------|
| car | 0.946 |
| motocycle | 0.905 |
| truck | 0.904 |
| bus | 0.795 |
| **Trung bình (all classes)** | **0.888** |

![Hình 5. Đường cong Precision–Recall](VehicleModel/vietnam_vehicle_v2/BoxPR_curve.png)

**Điểm F1–confidence (tham khảo triển khai):** trung bình mọi lớp đạt **F1 ≈ 0.84** tại **confidence ≈ 0.394** (cần hiệu chỉnh theo đánh đổi precision/recall).

![Hình 6. Đường cong F1–Confidence](VehicleModel/vietnam_vehicle_v2/BoxF1_curve.png)

![Hình 7. Precision–Confidence và Recall–Confidence](VehicleModel/vietnam_vehicle_v2/BoxP_curve.png)

![Hình 8. (tiếp) Recall–Confidence](VehicleModel/vietnam_vehicle_v2/BoxR_curve.png)

**Phân tích confusion matrix:** lớp **bus** có chỉ số thấp nhất trong nhóm; xuất hiện nhầm **bus → car** và **bus → background**. **False positive** nền → dự đoán `car` / `motocycle` gợi ý bổ sung mẫu nền âm hoặc điều chỉnh ngưỡng.

![Hình 9. Ma trận nhầm lẫn (chuẩn hóa)](VehicleModel/vietnam_vehicle_v2/confusion_matrix_normalized.png)

![Hình 10. Ma trận nhầm lẫn (số đếm)](VehicleModel/vietnam_vehicle_v2/confusion_matrix.png)

---

## 5. Phân đoạn người–đầu–mũ: YOLO hybrid *(đề xuất chính — đang thực hiện)*

### 5.1. Nhiệm vụ và nhãn

- Phân đoạn mask cho **`person`**, **`head`**, **`helmet`** để suy luận không đội mũ và số người (kết hợp tầng phương tiện phụ trợ).
- **Đầu vào:** ảnh toàn cảnh hoặc crop từ ROI do module §4 cung cấp.

**Lưu ý:** huấn luyện end-to-end, metric mAP/mask IoU và hình ảnh kết quả cho tầng này **sẽ được bổ sung** khi mô hình hybrid hoàn tất.

### 5.2. Synergy ViT–CBAM

| Thành phần | Vai trò |
|------------|--------|
| ViT (P5) | Ngữ cảnh toàn cục, hỗ trợ chi tiết khó |
| CBAM | Attention kênh–không gian, giảm nhiễu, biên mask rõ hơn |
| SPPF | Đa receptive field sau ViT, trước CBAM cuối backbone |

### 5.3. Quy tắc vi phạm (mẫu triển khai)

- **Không đội mũ:** ≥ *N* khung liên tiếp không phát hiện `helmet` trên người điều khiển (N cấu hình được).
- **Chở quá số người:** số instance `person` vượt ngưỡng so với loại phương tiện (ví dụ xe máy) do Model 1 xác định.

---

## 6. Dữ liệu và tổ chức thí nghiệm

### 6.1. Thu thập và tiền xử lý

- Video giao thông; trích frame theo tần số phù hợp.
- Lọc ảnh mờ (phương sai Laplacian), chuẩn hóa kích thước, tăng cường (lật, mờ, nhiễu).

### 6.2. Gán nhãn

Hai bài toán dùng chung nguồn ảnh có thể nhưng **tách thư mục nhãn**:

```
raw_images/
labels_model1/
labels_model2/
```

### 6.3. Huấn luyện

- Pretrained YOLOv8; có thể đóng băng backbone giai đoạn đầu rồi fine-tune head.
- Loss: tổn thất mặc định YOLO (box + cls + dfl); cộng tổn thất mask khi segmentation.
- **Hiện trạng:** baseline phương tiện (§4) đã có log đánh giá; **huấn luyện mô hình hybrid** (§3, §5) **đang tiến hành**, chưa gắn kết quả định lượng vào báo cáo cuối.

### 6.4. Chỉ số đánh giá

- **Phát hiện / phân đoạn:** Precision, Recall, mAP@0.5, mAP@0.5:0.95, IoU (mask).
- **Theo dõi:** MOTA, đổi ID, FPS.

---

## 7. Cơ chế ghi nhận vi phạm

- Chỉ lưu bằng chứng khi có vi phạm (trigger theo luật).
- Các loại lỗi độc lập; một phương tiện có thể ghi nhận nhiều vi phạm đồng thời.

---

## 8. Kết luận (Conclusion)

Đề tài hướng tới **kiến trúc phân đoạn hybrid YOLOv8–ViT–CBAM** cho ba lớp người–đầu–mũ — **đây là hướng nghiên cứu chính** và vẫn **đang triển khai**. Mô-đun phát hiện phương tiện chỉ đóng vai trò **phụ trợ** trong pipeline; số liệu mAP **0.888** @0.5 trên `vietnam_vehicle_v2` minh họa **baseline phụ**, không đại diện cho kết quả cuối của đề tài.

Việc tiếp theo: **hoàn thiện** huấn luyện và đánh giá mô hình hybrid, sau đó mới tổng hợp so sánh với baseline CNN thuần (nếu có). Đồng thời có thể tối ưu suy luận (TensorRT), mở rộng OCR biển số, và các hành vi phức tạp (sai làn, ngược chiều).

---

## 9. Hướng nghiên cứu tiếp theo

- **Ưu tiên:** kết thúc triển khai YOLO hybrid (YAML, loss mask, metric, ablation ViT/CBAM nếu cần).
- Tích hợp phát hiện biển số và OCR khi cần định danh phương tiện.
- Tối ưu realtime cho backbone hybrid trên thiết bị biên.
- Mở rộng tích hợp IoT / smart city.

---

## Tài liệu tham khảo (References)

1. Jocher, G., et al. *Ultralytics YOLOv8* (repository và tài liệu), 2023–2024.
2. Woo, S., et al. *CBAM: Convolutional Block Attention Module*, ECCV, 2018.
3. Dosovitskiy, A., et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR, 2021.
4. Lin, T.-Y., et al. *Microsoft COCO: Common Objects in Context*, ECCV, 2014 (bối cảnh metric mAP).

---

## Phụ lục: Công nghệ và môi trường

| Thành phần | Ghi chú |
|------------|--------|
| Ngôn ngữ | Python |
| Framework phát hiện / phân đoạn | YOLOv8 (Ultralytics) |
| Attention | CBAM; khối ViT trong backbone |
| Xử lý ảnh / video | OpenCV |
| Theo dõi | DeepSORT / ByteTrack |
