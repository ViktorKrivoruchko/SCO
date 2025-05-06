from ultralytics import YOLO

# Загрузка предобученной модели
model = YOLO('yolov8n.pt')  # Файл в корне SCO, путь указывать не нужно

# Запуск дообучения
model.train(
    data='SCO_products.v1i.yolov8/data.yaml',  # Путь к data.yaml
    epochs=50,
    imgsz=640,
    batch=16,
    name='custom_yolov8'
)

print("Обучение завершено! Модель сохранена в runs/detect/custom_yolov8/weights/best.pt")