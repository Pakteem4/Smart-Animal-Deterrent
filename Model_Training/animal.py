from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='data.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    workers=0,
    project='durian_yolo_runs',
    name='yolov8n_animals'
)

