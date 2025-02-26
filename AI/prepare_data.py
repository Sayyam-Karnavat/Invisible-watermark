import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import cv2

# Set directory paths
image_dir = "dataset/images/"
watermark_dir = "dataset/watermarks/"

os.makedirs(image_dir, exist_ok=True)
os.makedirs(watermark_dir, exist_ok=True)

# Load CIFAR-10 dataset
(x_train, _), (_, _) = cifar10.load_data()

# Resize & Save Images
for i, img in enumerate(x_train):
    resized_img = cv2.resize(img, (256, 256))  # Resize to 256x256
    img_path = os.path.join(image_dir, f"image_{i}.png")
    cv2.imwrite(img_path, resized_img)

    if i % 100 == 0:
        print(f"Saved {i}/1000 images...")

    if i == 1000:  # Save only first 1000 images
        break

print("All images saved successfully in dataset/images/")

# Generate & Save 100 Random Watermarks
for i in range(100):
    watermark = np.random.rand(256, 256, 1)  # Random grayscale watermark
    watermark_path = os.path.join(watermark_dir, f"watermark_{i}.npy")
    np.save(watermark_path, watermark)

print("100 watermarks saved successfully in dataset/watermarks/")
