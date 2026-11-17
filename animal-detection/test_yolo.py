import cv2
from pathlib import Path
from ultralytics import YOLO

# ตำแหน่งโฟลเดอร์โปรเจกต์
base = Path(__file__).resolve().parent

# โหลดโมเดล YOLOv8n ที่เทรนไว้
model_path = (
    base / "durian_yolo_runs" /
    "yolov8n_animals" /
    "weights" / "best.pt"
)

model = YOLO(str(model_path))

print("YOLOv8n model loaded successfully!")

# ตำแหน่งรูปภาพสำหรับทดสอบ
image_folder = (
    Path.home() / "Downloads" /
    "My First Project" / "test" / "images"
)

# ค้นหารูปภาพในโฟลเดอร์
image_extensions = ["*.jpg", "*.jpeg", "*.png"]
image_paths = []

for extension in image_extensions:
    image_paths.extend(image_folder.glob(extension))

if not image_paths:
    print("ไม่พบรูปภาพในโฟลเดอร์ test/images")
    exit()

print(f"Found {len(image_paths)} test images")

# วนตรวจจับภาพทีละภาพ
for image_path in image_paths:

    results = model.predict(
        source=str(image_path),
        conf=0.7,
        verbose=False
    )

    for result in results:

        # แสดงกรอบตรวจจับบนภาพ
        annotated_frame = result.plot()

        # แสดงผลลัพธ์
        cv2.imshow(
            "YOLOv8n Animal Detection",
            annotated_frame
        )

        # แสดงชื่อสัตว์ที่ตรวจพบ
        if len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                print(
                    f"Detected: {class_name} "
                    f"| Confidence: {confidence:.2f}"
                )
        else:
            print(f"No animal detected: {image_path.name}")

        print(f"Image: {image_path.name}")

        # กดปุ่มใดก็ได้เพื่อดูภาพถัดไป
        # กด q เพื่อหยุดโปรแกรม
        key = cv2.waitKey(0)

        if key == ord('q'):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
print("Finished testing!")
