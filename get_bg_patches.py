import cv2
import numpy as np
from pathlib import Path
import random


def extract_backgrounds_from_dataset(image_dir, label_dir, samples_per_image=5):
    """
    Extract background patches from each of your sample images
    Keep only the best ones
    """
    background_count = 0

    for img_path in Path(image_dir).glob('*.jpg'):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, cx, cy, bw, bh = map(float, parts[:5])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    boxes.append([x1, y1, x2, y2])
        f.close()
        
        for i in range(samples_per_image):

            for _ in range(10):  # 10 attempts

                
                crop_size = random.randint(128, min(w, h)//2)
                x = random.randint(0, max(0, w - crop_size))
                y = random.randint(0, max(0, h - crop_size))

                
                crop_box = [x, y, x + crop_size, y + crop_size]
                has_overlap = False

                for box in boxes:
                    
                    x_overlap = max(0, min(crop_box[2], box[2]) - max(crop_box[0], box[0]))
                    y_overlap = max(0, min(crop_box[3], box[3]) - max(crop_box[1], box[1]))
                    overlap_area = x_overlap * y_overlap
                    crop_area = crop_size * crop_size

                    if overlap_area > 0.1 * crop_area:  # Less than 10% overlap
                        has_overlap = True
                        break

                if not has_overlap:
                    # Create annotation file with class 3 (background)
                    with open(label_path, 'a') as f:
                        f.write("\n"+f"3 {round((x+crop_size/2)/w, 8)} {round((y+crop_size/2)/h, 8)} {round(crop_size/w, 8)} {round(crop_size/h, 8)}")
                    f.close()
                    background_count += 1
                    break

    print(f"Extracted {background_count} background patches")
    return background_count


extract_backgrounds_from_dataset(
    image_dir="train/images",
    label_dir="train/labels",
    samples_per_image=1
)