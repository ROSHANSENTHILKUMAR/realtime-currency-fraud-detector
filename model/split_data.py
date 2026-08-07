import splitfolders
splitfolders.ratio('dataset', output='split_dataset',
                    seed=42, ratio=(0.7, 0.15, 0.15))






import os
import shutil
import random

random.seed(42)

SOURCE_DIR = "dataset"
DEST_DIR = "split_dataset"
CLASSES = ["genuine", "fake"]
SPLIT_RATIO = 0.8  # 80% train, 20% val

for cls in CLASSES:
    src_folder = os.path.join(SOURCE_DIR, cls)
    files = os.listdir(src_folder)
    random.shuffle(files)

    split_point = int(len(files) * SPLIT_RATIO)
    train_files = files[:split_point]
    val_files = files[split_point:]

    train_dest = os.path.join(DEST_DIR, "train", cls)
    val_dest = os.path.join(DEST_DIR, "val", cls)
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(src_folder, f), os.path.join(train_dest, f))
    for f in val_files:
        shutil.copy(os.path.join(src_folder, f), os.path.join(val_dest, f))

    print(f"{cls}: {len(train_files)} train, {len(val_files)} val")

print("Split complete.")