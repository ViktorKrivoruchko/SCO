from ultralytics import YOLO
import os

# Проверка наличия файлов
model_path = 'yolov8n.pt'
data_path = 'SCO_products.v1i.yolov8/data.yaml'

if not os.path.exists(model_path):
    print(f"Ошибка: файл {model_path} не найден!")
    exit(1)

if not os.path.exists(data_path):
    print(f"Ошибка: файл {data_path} не найден! Проверьте путь относительно {os.getcwd()}")
    exit(1)

# Загрузка предобученной модели
model = YOLO(model_path)

# Запуск дообучения
model.train(
    data=data_path,
    epochs=200,  # Увеличено до 200 эпох
    imgsz=640,   # Соответствует обучению
    batch=16,    # Согласно args.yaml
    name='custom_yolo811',
    augment=True,  # Включить аугментацию
    patience=20,
    lr0=0.0005,   # Уменьшить скорость обучения
    momentum=0.937,
    weight_decay=0.0005
)

print("Обучение завершено! Модель сохранена в runs/detect/custom_yolo811/weights/best.pt")