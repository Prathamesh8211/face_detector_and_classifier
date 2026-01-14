import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(BASE_DIR, "data", "cropped")
TARGET_DIR = os.path.join(BASE_DIR, "data", "dataset")

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(42)

for category in os.listdir(SOURCE_DIR):
    category_path = os.path.join(SOURCE_DIR, category)
    images = os.listdir(category_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * SPLIT_RATIO["train"])
    val_end = train_end + int(total * SPLIT_RATIO["val"])

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        dest_dir = os.path.join(TARGET_DIR, split, category)
        os.makedirs(dest_dir, exist_ok=True)

        for file in files:
            src = os.path.join(category_path, file)
            dst = os.path.join(dest_dir, file)
            shutil.copy(src, dst)

print("[DONE] Dataset split completed.")
