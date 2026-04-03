# SystemTrafficLaw

## Hệ thống phát hiện vi phạm giao thông với YOLO26-Hybrid (ViT + CBAM)

README này đã được cập nhật theo đúng kiến trúc bạn cung cấp: YOLO26-Hybrid với TransformerBlock (ViT) và CBAM cho bài toán segmentation `person/head/helmet`.

---

## 1. Tổng quan bài toán

Hệ thống xử lý video giao thông theo mô hình 2 tầng:

1. Tầng phát hiện và tracking phương tiện (`Vehicle.pt`) để lấy ROI và track ID.
2. Tầng segmentation hybrid (`ViTs+CBAM.pt`) để suy luận vi phạm mũ bảo hiểm trên vùng xe máy.

Mục tiêu cuối là phát hiện vi phạm theo thời gian thực, gồm:

- Vượt đèn đỏ (ROI đèn + vạch dừng ảo + tracking).
- Không đội mũ bảo hiểm (segmentation `person/head/helmet`).

---

## 2. Kiến trúc YOLO26-Hybrid

### 2.1 Sơ đồ tổng thể Backbone - Neck - Head

![YOLO26-Hybrid Architecture](image/YOLO26HYBRID.png)

### 2.2 CBAM module (Channel + Spatial Attention)

![CBAM Architecture](image/ArchitectureCBAM.png)

### 2.3 TransformerBlock (ViT block)

![TransformerBlock Architecture](image/ViTs.png)

---

## 3. Ánh xạ kiến trúc vào YAML hiện tại

File cấu hình: `yolo26seg_cbam_vits.yaml`

### Backbone

- Stage đầu: `Conv -> Conv -> C3k2` để trích xuất đặc trưng cơ bản.
- P3 (stride 8): `Conv -> C3k2 -> CBAM`.
- P4 (stride 16): `Conv -> C3k2 -> CBAM`.
- P5 (stride 32): `Conv -> C3k2 -> SPPF -> TransformerBlock`.

### Neck

- Nhánh top-down kiểu FPN: `Upsample + Concat + C3k2`.
- Nhánh bottom-up kiểu PAN: `Conv + Concat + C3k2`.

### Head

- Head segmentation đa tỉ lệ qua 3 mức đặc trưng `[18, 21, 24]`.
- Cấu hình hiện tại: `Segment[nc=3, 32, 3]`.
- Nhãn segmentation mục tiêu: `person`, `head`, `helmet`.

---

## 4. Vai trò từng thành phần

- `CBAM`: tăng trọng số vào kênh và vùng không gian quan trọng, giảm nhiễu nền.
- `TransformerBlock`: bổ sung ngữ cảnh toàn cục ở mức đặc trưng sâu (P5).
- `SPPF`: mở rộng receptive field trước khi đưa vào phần fusion/head.
- `Segment head`: sinh mask cho 3 lớp phục vụ luật vi phạm.

---

## 5. Pipeline xử lý vi phạm

```text
Video Input
   -> Vehicle Detection + Tracking (Vehicle.pt)
   -> Rule Engine (đèn đỏ, vạch dừng, ROI)
   -> Segmentation on motorcycle ROI (ViTs+CBAM.pt)
   -> Violation Decision (no-helmet / red-light)
   -> Lưu video kết quả + evidence
```

Script chính đang chạy pipeline: `traffic_hybrid_system.py`

---

## 6. Cấu trúc project (rút gọn)

```text
SystemTrafficLaw/
├─ traffic_hybrid_system.py
├─ yolo26seg_cbam_vits.yaml
├─ scripts/
│  ├─ CBAM.py
│  └─ Transformer.py
├─ models/
│  ├─ Vehicle.pt
│  └─ ViTs+CBAM.pt
├─ image/
│  ├─ YOLO26HYBRID.png
│  ├─ ArchitectureCBAM.png
│  └─ ViTs.png
└─ src/
```

---

## 7. Cài đặt nhanh

```bash
pip install -r requirements.txt
```

Lưu ý:

- Cần đặt đúng weights tại `models/Vehicle.pt` và `models/ViTs+CBAM.pt`.
- Dự án dùng `ultralytics`, `torch`, `opencv-python`.

---

## 8. Chạy hệ thống

```bash
python traffic_hybrid_system.py --video path/to/video.mp4 --out out.mp4
```

Hoặc chạy chế độ hiển thị trực tiếp:

```bash
python traffic_hybrid_system.py --video path/to/video.mp4 --show
```

---

## 9. Trạng thái hiện tại

- Đã có kiến trúc YOLO26-Hybrid hoàn chỉnh ở mức thiết kế và tích hợp module.
- Đã có mã CBAM và TransformerBlock riêng trong thư mục `scripts/`.
- Đang tiếp tục tinh chỉnh huấn luyện/evaluate cho segmentation 3 lớp theo kiến trúc hybrid.

---

## 10. Hướng phát triển tiếp

1. Ablation study: so sánh baseline YOLO thuần với YOLO26-Hybrid (ViT + CBAM).
2. Tối ưu tốc độ suy luận thời gian thực (TensorRT/ONNX).
3. Bổ sung OCR biển số và đồng bộ với backend báo cáo vi phạm.

---

## Tài liệu liên quan

- `QUICKSTART.md`: hướng dẫn chạy nhanh cho các module tiện ích.
- `Document/Hybrid.md`: ghi chú chuyên sâu về kiến trúc hybrid (nếu có cập nhật).
