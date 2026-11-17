import cv2
import numpy as np
import urllib.request
from pathlib import Path
from ultralytics import YOLO

# โหลดโมเดล YOLOv8n
base = Path(__file__).resolve().parent


model_path = (
    base / "durian_yolo_runs" /
    "yolov8n_animals" /
    "weights" / "best.pt"
)

model = YOLO(str(model_path))

print("YOLOv8n model loaded successfully!")

# ตั้งค่ากล้อง ESP32-CAM

url = 'http://192.168.1.187/cam-hi.jpg'
confThreshold = 0.7

# เริ่มรับภาพและตรวจจับสัตว์
while True:

    try:
        # รับภาพจาก ESP32-CAM
        img_resp = urllib.request.urlopen(url, timeout=5)

        imgnp = np.array(
            bytearray(img_resp.read()),
            dtype=np.uint8
        )

        frame = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)

        if frame is None:
            print("ไม่สามารถอ่านภาพจากกล้องได้")
            continue

    except Exception as e:
        print("ไม่สามารถรับภาพจาก ESP32-CAM ได้:", e)
        continue

    # วิเคราะห์ภาพด้วย YOLOv8n
    results = model.predict(
        source=frame,
        conf=confThreshold,
        verbose=False
    )

    detected = False

    for result in results:
        for box in result.boxes:
            # อ่านพิกัดกรอบตรวจจับ
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )
            # อ่านชื่อคลาสและค่าความมั่นใจ
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            detected = True

            # วาดกรอบรอบสัตว์
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # แสดงชื่อสัตว์และค่าความมั่นใจ
            cv2.putText(
                frame,
                f"{class_name} ({confidence:.2f})",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            print(
                f"ALERT: {class_name.upper()} detected! "
                f"Confidence: {confidence:.2f}"
            )

    # แสดงสถานะการตรวจจับ
    if detected:

        cv2.putText(
            frame,
            "ANIMAL DETECTED!",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "SAFE!",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # แสดงภาพจากกล้อง
    cv2.imshow('YOLOv8n Animal Detection', frame)

    # กด q เพื่อหยุดโปรแกรม
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
