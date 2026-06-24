import cv2
import os
import numpy as np
import random

DATASET = "Dataset_filtrado"
OUTPUT = "Dataset_aug"

os.makedirs(OUTPUT, exist_ok=True)

def augment(img):

    h, w = img.shape[:2]

    # 1. brillo / contraste leve
    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-10, 10)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 2. blur leve (simula movimiento de mano)
    if random.random() < 0.3:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    # 3. pequeño shift (pizarra no perfecta)
    if random.random() < 0.4:
        tx = random.randint(-3, 3)
        ty = random.randint(-3, 3)

        M = np.float32([[1, 0, tx], [0, 1, ty]])
        out = cv2.warpAffine(out, M, (w, h), borderValue=255)

    # 4. zoom leve
    if random.random() < 0.3:
        scale = random.uniform(0.9, 1.05)
        M = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
        out = cv2.warpAffine(out, M, (w, h), borderValue=255)

    return out


for cls in os.listdir(DATASET):

    in_folder = os.path.join(DATASET, cls)
    out_folder = os.path.join(OUTPUT, cls)

    os.makedirs(out_folder, exist_ok=True)

    idx = 0

    for file in os.listdir(in_folder):

        path = os.path.join(in_folder, file)
        img = cv2.imread(path)

        if img is None:
            continue

        # guardar original también
        cv2.imwrite(os.path.join(out_folder, f"{cls}_{idx}.jpg"), img)
        idx += 1

        # generar N augmentations por imagen
        for _ in range(2):   # 2x expansión (puedes subir a 3 si quieres)
            aug = augment(img)
            cv2.imwrite(os.path.join(out_folder, f"{cls}_{idx}.jpg"), aug)
            idx += 1

    print(f"{cls} listo -> {idx} imágenes")