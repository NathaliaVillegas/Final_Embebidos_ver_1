from ultralytics import YOLO

# 1. modelo base preentrenado
model = YOLO("yolov8n.pt")

# 2. entrenar con tu dataset
model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)

# 3. exportar modelo entrenado (opcional pero recomendado)
model.export(format="onnx")   # o "tflite" si quieres Raspberry Pi optimizado