import os
from ddgs import DDGS
import requests


NEGATIVE_KEYWORDS = [
	"empty checkpoint booth india",
	"indian road without people",
	"military vehicle interior empty",
	"barracks room empty",
	"camouflage pattern fabric",
	"khaki colored wall",
	"olive green tarp",
	"indian civilian crowd",
	"construction workers india",
	"security guard uniform india",
	"kashmir landscape empty",
	"border fence india",
	"military base building exterior"
]
MAX_IMAGES_PER_KEYWORD = 4         # per keyword
NEGATIVE_RATIO = 0.25              # negatives as % of total positives
YOLO_DATASET_DIR = "train/images"  # path to your YOLO dataset root


def download_image(url, path):
    try:
        img_data = requests.get(url, timeout=10).content
        with open(path, 'wb') as handler:
            handler.write(img_data)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


# Paths
max_negatives = NEGATIVE_RATIO*len(os.listdir(YOLO_DATASET_DIR)) 
downloaded_count = 0

with DDGS() as ddgs:
    for keyword in NEGATIVE_KEYWORDS:
        if downloaded_count >= max_negatives:
            break
        print(f"Searching: {keyword}")
        results = ddgs.images(
            keyword,               
            region="in-en",
            safesearch="off",
            size=None,
            type_image=None,
            layout=None,
            license_image=None,
            max_results=MAX_IMAGES_PER_KEYWORD
        )
        for idx, img in enumerate(results):
            if downloaded_count >= max_negatives:
                break
            url = img.get("image")
            if not url:
                continue

            filename = f"neg_{downloaded_count:04d}.jpg"
            img_path = os.path.join(YOLO_DATASET_DIR, filename)

            if download_image(url, img_path):
                with open(os.path.join('train', 'labels', filename[:-4]+'.txt'), 'w') as f:
                    f.write('BG 0.5 0.5 1 1')
                f.close()
                downloaded_count += 1
                print(f"Saved : {img_path}")

print(f"\nAdded {downloaded_count} negative images.")

