import cv2
import os
import re
import time

# =========================
# CONFIG
# =========================

SAVE_DIR = "Dataset_math"

classes = {
    ord('0'): "0",
    ord('1'): "1",
    ord('2'): "2",
    ord('3'): "3",
    ord('4'): "4",
    ord('5'): "5",
    ord('6'): "6",
    ord('7'): "7",
    ord('8'): "8",
    ord('9'): "9",

    ord('+'): "plus",
    ord('-'): "minus",
    ord('*'): "multiply",
    ord('/'): "divide",
    ord('='): "equal",

    ord('('): "open_paren",
    ord(')'): "close_paren"
}

# =========================
# CREAR CARPETAS
# =========================

for cls in classes.values():
    os.makedirs(os.path.join(SAVE_DIR, cls), exist_ok=True)

# =========================
# NEXT INDEX SEGURO
# =========================

def next_index(folder, prefix):
    max_num = 0

    for file in os.listdir(folder):
        match = re.match(rf"{prefix}_(\d+)\.jpg", file)
        if match:
            max_num = max(max_num, int(match.group(1)))

    return max_num + 1

# =========================
# CAPTURA MODO RÁFAGA
# =========================

def capture_burst(frame, folder, prefix, n=20):
    start_idx = next_index(folder, prefix)

    for i in range(n):
        idx = start_idx + i
        filename = f"{prefix}_{idx}.jpg"
        path = os.path.join(folder, filename)

        cv2.imwrite(path, frame)
        print(f"Guardado: {path}")

        time.sleep(0.05)

# =========================
# CAMARA
# =========================

cap = cv2.VideoCapture(2)
cv2.namedWindow("Camara", cv2.WINDOW_NORMAL)

print("""
CONTROLES:
0-9 + - * / = ( ) -> guardar imagen
r -> ráfaga (20 imágenes)
q -> salir
""")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ===== IMPORTANTE =====
    # SI quieres rotar como lo ves físicamente:
    display = frame

    # (IMPORTANTE: lo que se guarda es LO MISMO que ves)
    save_frame = display.copy()

    cv2.putText(display,
                "0-9 + - * / = ( ) | r burst | q quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2)

    cv2.imshow("Camara", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # =========================
    # CAPTURA NORMAL
    # =========================
    if key in classes:
        cls = classes[key]
        folder = os.path.join(SAVE_DIR, cls)

        idx = next_index(folder, cls)
        filename = f"{cls}_{idx}.jpg"

        path = os.path.join(folder, filename)
        cv2.imwrite(path, save_frame)

        print(f"Guardado: {path}")

    # =========================
    # RÁFAGA (CLAVE PARA VARIABILIDAD REAL)
    # =========================
    if key == ord('r'):
        # debes primero presionar un símbolo antes de r
        print("Selecciona clase antes de ráfaga")

    # versión mejor: detección simple
    # (mantiene último símbolo presionado)
    if key in classes:
        last_cls = classes[key]

    if key == ord('r') and 'last_cls' in locals():
        folder = os.path.join(SAVE_DIR, last_cls)
        capture_burst(save_frame, folder, last_cls, n=20)

cap.release()
cv2.destroyAllWindows()