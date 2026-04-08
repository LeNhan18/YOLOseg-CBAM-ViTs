"""
Test nhanh giống kiểu cũ: mở video → xử lý từng frame → hiển thị.
Dùng 2 weight: models/Vehicle.pt + models/ViTs+CBAM.pt (đèn đỏ + không mũ trên xe máy).

Đổi video_path / đường dẫn model ở dưới. ESC thoát.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import patch_ultralytics_custom_loss  # noqa: F401 — load seg .pt có BCEDiceLoss
except ImportError:
    pass

from traffic_hybrid_system import HybridConfig, TrafficHybridSystem

VIDEO_PATH = r"D:\nl\DatasetViPhamGiaoThong-20260211T113245Z-3-001\DatasetViPhamGiaoThong\Traffic063.mp4"
VEHICLE_PT = _ROOT / "models" / "Vehicle.pt"
SEG_PT = _ROOT / "models" / "ViTs+CBAM.pt"


cfg = HybridConfig(
    vehicle_weights=VEHICLE_PT,
    seg_weights=SEG_PT,
    seg_only_mode=True,        # Tạm tắt Vehicle.pt, chỉ chạy ViTs+CBAM
    conf_seg=0.35,             # Ngưỡng detect — tăng nếu vẫn quá nhiều nhiễu
    max_mask_area_ratio=0.20,  # Bỏ qua mask bbox chiếm > 20% frame
    debug_log_seg=True,        # In class nào đang detect ra console
)

sys_model = TrafficHybridSystem(cfg)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Không mở được video:", VIDEO_PATH)
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Hết video hoặc lỗi frame")
        break

    try:
        annotated, viol = sys_model.process_frame(frame)
    except Exception as e:
        print("Lỗi frame, skip:", e)
        continue

    if viol.messages:
        cv2.putText(
            annotated,
            viol.messages[-1][:60],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    cv2.imshow("TrafficHybrid (ESC thoat)", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
