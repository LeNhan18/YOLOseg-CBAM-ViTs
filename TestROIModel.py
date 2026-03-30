from ultralytics import YOLO
import cv2
import numpy as np

# Load model
model = YOLO("models\\best.pt")

# Đường dẫn ảnh
image_path = r"D:\nl\DatasetViPhamGiaoThong-20260211T113245Z-3-001\DatasetViPhamGiaoThong\Traffic065.PNG"
frame = cv2.imread(image_path)
height, width = frame.shape[:2]

# ================== ĐIỀU CHỈNH THEO ẢNH CỦA BẠN ==================
# Vạch dừng ảo: đặt sát vạch kẻ đường thật (zebra crossing) ở dưới cùng
STOP_LINE_Y = int(height * 0.78)  # ~78% chiều cao từ trên xuống (thử 0.75 đến 0.85)
STOP_LINE_COLOR = (0, 255, 255)
LINE_THICKNESS = 4

# Vùng ROI đèn giao thông (hard-code theo vị trí cố định trong video/ảnh của bạn)
# Trong ảnh: đèn nằm ở phần trên giữa, khoảng từ y=0 đến y=200, x giữa
LIGHT_ROI = (width // 2 - 300, 0, width // 2 + 300, 250)  # (x1, y1, x2, y2) - điều chỉnh nếu cần


# ================== HÀM CHECK ĐÈN ĐỎ ==================
def is_red_light_in_roi(frame, roi):
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Phạm vi đỏ chặt hơn để tránh nhầm
    lower_red1 = np.array([0, 140, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 140, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    red_ratio = np.sum(mask > 0) / mask.size
    return red_ratio > 0.18  # ngưỡng chặt hơn


# ================== XỬ LÝ ==================
results = model.track(frame, persist=True, conf=0.35, iou=0.5, tracker="bytetrack.yaml")
annotated = results[0].plot(line_width=2)

# Vẽ vạch dừng ảo
cv2.line(annotated, (0, STOP_LINE_Y), (width, STOP_LINE_Y), STOP_LINE_COLOR, LINE_THICKNESS)
cv2.putText(annotated, "VIRTUAL STOP LINE", (50, STOP_LINE_Y - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, STOP_LINE_COLOR, 3)

# Kiểm tra đèn đỏ trong ROI cố định
red_detected = is_red_light_in_roi(frame, LIGHT_ROI)

if red_detected:
    cv2.putText(annotated, "DEN DO!", (width // 2 - 150, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)

    # Vẽ khung ROI đèn để debug
    x1, y1, x2, y2 = LIGHT_ROI
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Check vi phạm: xe vượt vạch khi đèn đỏ
    for box in results[0].boxes:
        if box.id is None: continue
        cls = int(box.cls)
        label = model.names[cls]

        if label in ['motorbike', 'car', 'bus', 'truck', 'person']:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            track_id = int(box.id)

            if xyxy[3] > STOP_LINE_Y:  # bottom vượt vạch
                cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 5)
                cv2.putText(annotated, f"VUOT DEN DO! ID:{track_id}",
                            (xyxy[0], xyxy[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

# Hiển thị & lưu
cv2.imshow("Corrected Detection", annotated)
cv2.waitKey(0)
cv2.imwrite(image_path.replace(".jpg", "_fixed.jpg"), annotated)
cv2.destroyAllWindows()