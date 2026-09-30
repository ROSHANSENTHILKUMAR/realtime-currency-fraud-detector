import cv2
import os
import numpy as np

input_dir = "dataset/not_currency"
output_dir = "dataset/not_currency"  # saves augmented versions in the same folder

images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(images)} original images. Starting augmentation...")

count = 0
for img_name in images:
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    base_name = os.path.splitext(img_name)[0]

    # 1. Horizontal flip
    flipped = cv2.flip(img, 1)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_flip.jpg"), flipped)
    count += 1

    # 2. Brightness increase
    bright = cv2.convertScaleAbs(img, alpha=1.0, beta=40)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_bright.jpg"), bright)
    count += 1

    # 3. Brightness decrease
    dark = cv2.convertScaleAbs(img, alpha=1.0, beta=-40)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_dark.jpg"), dark)
    count += 1

    # 4. Slight rotation
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w/2, h/2), 15, 1)
    rotated = cv2.warpAffine(img, matrix, (w, h))
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_rot.jpg"), rotated)
    count += 1

print(f"Done. Created {count} new augmented images.")
print("Total images now in folder:", len(os.listdir(input_dir)))