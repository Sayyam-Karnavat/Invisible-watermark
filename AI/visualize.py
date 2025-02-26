import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the trained encoder model
encoder = tf.keras.models.load_model("encoder_model.keras")

# Function to load and preprocess images
def load_image(image_path, target_size=(256, 256)):
    """Loads and preprocesses an image."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize to [0,1]
    return img.astype(np.float32)

# Function to generate random watermark
def generate_watermark(size=(32, 32)):
    """Generates a random 32x32 grayscale watermark."""
    return np.random.rand(*size, 1).astype(np.float32)

# Load test image
image_path = "dataset/original.png"  # Change this to your test image path
cover_image = load_image(image_path)  # Shape: (256, 256, 3)

# Generate test watermark
watermark = generate_watermark()  # Shape: (32, 32, 1)

# Expand dimensions to match model input
cover_image_batch = np.expand_dims(cover_image, axis=0)  # (1, 256, 256, 3)
watermark_batch = np.expand_dims(watermark, axis=0)  # (1, 32, 32, 1)

# Encode the image
encoded_image_batch = encoder.predict([cover_image_batch, watermark_batch])

# Remove batch dimension
encoded_image = np.squeeze(encoded_image_batch, axis=0)  # Shape: (256, 256, 3)

# Convert images back to [0,255] range for visualization
cover_image_display = (cover_image * 255).astype(np.uint8)
encoded_image_display = (encoded_image * 255).astype(np.uint8)

# Show original and encoded images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cover_image_display, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(encoded_image_display, cv2.COLOR_BGR2RGB))
plt.title("Encoded Image (with watermark)")
plt.axis("off")

plt.show()
