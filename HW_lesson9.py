import cv2
import os
from collections import defaultdict
import numpy as np

net = cv2.dnn.readNetFromCaffe('data/MobileNet/mobilenet_deploy.prototxt', 'data/MobileNet/mobilenet.caffemodel')

classes = []
with open('data/MobileNet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        name = name.replace("'", "")
        classes.append(name)

IMAGE_FOLDER = 'images/MobileNet'
class_counts = defaultdict(int)
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.jpg')]
result_images = []
DISPLAY_SIZE = (400, 400)

for filename in image_files:
    image_path = os.path.join(IMAGE_FOLDER, filename)
    image = cv2.imread(image_path)

    if image is None:
        continue

    original_image_for_display = image.copy()

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))

    net.setInput(blob)
    preds = net.forward()

    idx = preds[0].argmax()

    label = classes[idx] if idx < len(classes) else "unknown"
    conf = float(preds[0][idx]) * 100

    class_counts[label] += 1

    text = label + ": " + str(int(conf)) + "%"

    resized_image = cv2.resize(original_image_for_display, DISPLAY_SIZE)
    cv2.putText(resized_image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    result_images.append(resized_image)

if result_images:

    row_images = []

    if len(result_images) <= 3:
        row_images = result_images
    else:
        mid = (len(result_images) + 1) // 2
        top_row = result_images[:mid]
        bottom_row = result_images[mid:]

        if len(top_row) != len(bottom_row):
            blank_image = np.zeros((*DISPLAY_SIZE, 3), np.uint8)
            bottom_row.append(blank_image)

        top_combined = np.hstack(top_row)
        bottom_combined = np.hstack(bottom_row)

        row_images = [top_combined, bottom_combined]

    if len(row_images) == 1:
        final_image = row_images[0]
    else:
        final_image = np.vstack(row_images)

    print("Підрахунок зустрічальності класів:")
    for label, count in class_counts.items():
        print(f"{label}: {count}")

    cv2.imshow("Результати класифікації", final_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()