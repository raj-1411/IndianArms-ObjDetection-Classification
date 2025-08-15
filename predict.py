import os
from ultralytics import YOLO
from PIL import Image
import argparse


parser = argparse.ArgumentParser(description="Run YOLO11 inference on test images")
parser.add_argument('--model_weights', type=str, default='weights/best.pt', help='Path to the YOLO model weights')
parser.add_argument('--test_images_folder', type=str, default='Lab\'s/BSF', help='Folder containing test images')
parser.add_argument('--results_folder', type=str, default='Lab\'s/results/BSF', help='Folder to save results')
args = parser.parse_args()

model_weights_path = args.model_weights
test_images_folder = args.test_images_folder
results_folder = args.results_folder

model = YOLO(model_weights_path)

os.makedirs(results_folder, exist_ok=True)

image_files = [f for f in os.listdir(test_images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(test_images_folder, image_file)
    results = model(image_path)
    pred_image_path_temp = os.path.join(results_folder, f'temp_pred_{image_file}')
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save(pred_image_path_temp)

    print(f"Processed {image_file}")

print("Processing complete.")
