import cv2
import os
import numpy as np

DATASET = "Dataset_math"
BAD_FOLDER = "Dataset_removed_bad"
os.makedirs(BAD_FOLDER, exist_ok=True)

def blur_score(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def contrast_score(img):
    return img.std()

for cls in os.listdir(DATASET):

    folder = os.path.join(DATASET, cls)
    bad_cls_folder = os.path.join(BAD_FOLDER, cls)
    os.makedirs(bad_cls_folder, exist_ok=True)

    files = [f for f in os.listdir(folder) if f.endswith(".jpg")]

    print(f"\nClase: {cls}")

    scores = []

    for f in files:

        path = os.path.join(folder, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        b = blur_score(img)
        c = contrast_score(img)

        scores.append((f, b, c))

    # =========================
    # thresholds (ajustables)
    # =========================

    blur_threshold = 60      # debajo = borroso
    contrast_min = 20        # muy plano
    contrast_max = 90        # demasiado fuerte artificial

    removed = 0

    for f, b, c in scores:

        if b < blur_threshold or c < contrast_min or c > contrast_max:

            src = os.path.join(folder, f)
            dst = os.path.join(bad_cls_folder, f)

            os.rename(src, dst)
            removed += 1

    print(f"Eliminadas por calidad: {removed}")