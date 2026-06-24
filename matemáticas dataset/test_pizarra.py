from ultralytics import YOLO
import cv2

# cargar TU modelo ya entrenado
model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    symbols = []

    # extraer detecciones
    for box in results.boxes:
        cls = int(box.cls[0])
        x = float(box.xyxy[0][0])  # coordenada X

        symbols.append((x, cls))

    # ordenar de izquierda a derecha (clave para ecuaciones)
    symbols.sort(key=lambda s: s[0])

    # reconstruir expresión
    expression = "".join(str(s[1]) for s in symbols)

    print("Expresión detectada:", expression)

    # mostrar visualización
    cv2.imshow("YOLO Pizarra", results.plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()