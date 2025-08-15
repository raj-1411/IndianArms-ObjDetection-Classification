import os
from ultralytics import YOLO
from PIL import Image


model_weights_path = 'weights/best.pt'  
catg = 'BSF' # BSF, J&K or CRPF
test_images_folder = f'../Lab\'s/{catg}'
results_folder = f'Lab\'s/results/{catg}'

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