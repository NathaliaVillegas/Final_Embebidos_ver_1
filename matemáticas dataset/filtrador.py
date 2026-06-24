import os
import cv2
import shutil
import imagehash
import numpy as np

from PIL import Image
from tqdm import tqdm

# ==================================================
# CONFIG
# ==================================================

INPUT_ROOT = "Dataset_aug"
OUTPUT_ROOT = "Dataset_filtrado_final"

TARGET_IMAGES = 500

HASH_THRESHOLD = 1

# ==================================================
# CREAR DESTINO
# ==================================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ==================================================
# FEATURE PARA DIVERSIDAD
# ==================================================

def extract_feature(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(img, (64, 64))

    img = img.astype(np.float32)

    return img.flatten()

# ==================================================
# ELIMINAR DUPLICADOS
# ==================================================

def remove_duplicates(paths):

    selected = []
    hashes = []

    for path in tqdm(paths, leave=False):

        try:
            ph = imagehash.phash(Image.open(path))
        except:
            continue

        duplicate = False

        for h in hashes:

            if abs(ph - h) <= HASH_THRESHOLD:
                duplicate = True
                break

        if not duplicate:
            hashes.append(ph)
            selected.append(path)

    return selected

# ==================================================
# FARTHEST POINT SAMPLING
# ==================================================

def select_diverse(paths, target):

    if len(paths) <= target:
        return paths

    features = []

    valid_paths = []

    print("Extrayendo características...")

    for p in tqdm(paths):

        feat = extract_feature(p)

        if feat is not None:
            features.append(feat)
            valid_paths.append(p)

    features = np.array(features)

    n = len(valid_paths)

    if n <= target:
        return valid_paths

    print("Aplicando selección diversa...")

    selected_idx = [0]

    distances = np.linalg.norm(
        features - features[0],
        axis=1
    )

    for _ in tqdm(range(target - 1)):

        farthest = np.argmax(distances)

        selected_idx.append(farthest)

        new_dist = np.linalg.norm(
            features - features[farthest],
            axis=1
        )

        distances = np.minimum(
            distances,
            new_dist
        )

    return [valid_paths[i] for i in selected_idx]

# ==================================================
# PROCESAR CADA CLASE
# ==================================================

classes = sorted(os.listdir(INPUT_ROOT))

for cls in classes:

    class_dir = os.path.join(INPUT_ROOT, cls)

    if not os.path.isdir(class_dir):
        continue

    print("\n===================================")
    print("Clase:", cls)
    print("===================================")

    output_dir = os.path.join(OUTPUT_ROOT, cls)

    os.makedirs(output_dir, exist_ok=True)

    image_paths = []

    for file in os.listdir(class_dir):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            image_paths.append(
                os.path.join(class_dir, file)
            )

    print("Originales:", len(image_paths))

    # ----------------------------------
    # eliminar duplicados
    # ----------------------------------

    image_paths = remove_duplicates(image_paths)

    print("Tras eliminar duplicados:", len(image_paths))

    # ----------------------------------
    # seleccionar diversas
    # ----------------------------------

    selected = select_diverse(
        image_paths,
        TARGET_IMAGES
    )

    print("Seleccionadas:", len(selected))

    # ----------------------------------
    # guardar originales
    # ----------------------------------

    for idx, src in enumerate(selected, start=1):

        ext = os.path.splitext(src)[1]

        dst = os.path.join(
            output_dir,
            f"{cls}_{idx}{ext}"
        )

        shutil.copy2(src, dst)

print("\nProceso terminado.")