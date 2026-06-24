import os
import random
import shutil

DATASET = "Dataset_filtrado_final"

IMAGES = os.path.join(DATASET, "images")
LABELS = os.path.join(DATASET, "labels")

OUT_IMG_TRAIN = IMAGES + "/train"
OUT_IMG_VAL = IMAGES + "/val"

OUT_LBL_TRAIN = LABELS + "/train"
OUT_LBL_VAL = LABELS + "/val"

for p in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_LBL_TRAIN, OUT_LBL_VAL]:
    os.makedirs(p, exist_ok=True)

images = [f for f in os.listdir(IMAGES) if f.endswith(".jpg") or f.endswith(".png")]

random.shuffle(images)

split = int(len(images) * 0.8)

train_imgs = images[:split]
val_imgs = images[split:]

def move_files(img_list, mode):
    for img in img_list:
        label = img.replace(".jpg", ".txt").replace(".png", ".txt")

        if mode == "train":
            shutil.move(os.path.join(IMAGES, img), OUT_IMG_TRAIN)
            if os.path.exists(os.path.join(LABELS, label)):
                shutil.move(os.path.join(LABELS, label), OUT_LBL_TRAIN)

        else:
            shutil.move(os.path.join(IMAGES, img), OUT_IMG_VAL)
            if os.path.exists(os.path.join(LABELS, label)):
                shutil.move(os.path.join(LABELS, label), OUT_LBL_VAL)

move_files(train_imgs, "train")
move_files(val_imgs, "val")

print("Dataset separado correctamente")