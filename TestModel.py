from ultralytics import YOLO
import cv2
import os
model_path  = r"D:\nl\best (1).pt"
model =YOLO(model_path)
video_path = r"D:\nl\DatasetViPhamGiaoThong-20260211T113245Z-3-001\DatasetViPhamGiaoThong\Traffic059.mp4"
cap = cv2.VideoCapture(video_path)
while True:                             
    ret, frame = cap.read()
    if not ret:
        print("Hết video hoặc lỗi frame")
        break

    try:
        results = model.track(frame, persist=True)
        annotated = results[0].plot()
    except Exception as e:
        print("Lỗi frame, skip:", e)
        continue

    cv2.imshow("Tracking", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()

cv2.destroyAllWindows()