import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import qrcode
from PIL import Image
import random

# Set OneDNN flag for TensorFlow optimizations


# Define paths
dataset_path = "dataset/original_images"       # Folder containing original images
watermark_path = "dataset/qr_watermarks"         # Folder to save generated QR codes
augmented_path = "dataset/augmented_images"      # Folder to save augmented images

# Create necessary directories if they don't exist
os.makedirs(augmented_path, exist_ok=True)
os.makedirs(watermark_path, exist_ok=True)

# Image augmentation configuration (applied to cover images)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode="nearest"
)

# ----------------------- Function to Augment Cover Images -----------------------
def load_and_augment_images(dataset_path, save_path, target_size=(256, 256), num_augmented=10):
    """
    Loads images from dataset_path, applies augmentation, and saves them to save_path.
    """
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, target_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)  # Shape: (1, 256, 256, 3)
            
            # Generate augmented images
            aug_iter = datagen.flow(image, batch_size=1)
            for i in range(num_augmented):
                augmented_image = next(aug_iter)[0]
                augmented_image = (augmented_image * 255).astype(np.uint8)
                save_filename = os.path.join(save_path, f"{os.path.splitext(filename)[0]}_aug{i}.png")
                cv2.imwrite(save_filename, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

# ----------------------- Function to Generate Pool of QR Codes -----------------------
def generate_qr_codes(data_list, save_path, size=256):
    """
    Generates QR code images from a list of text data and saves them as grayscale images.
    """
    qr_codes = []
    for i, data in enumerate(data_list):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("L")
        img = img.resize((size, size))
        qr_array = np.array(img, dtype=np.float32) / 255.0
        qr_codes.append(qr_array)
        save_filename = os.path.join(save_path, f"qr_{i}.png")
        cv2.imwrite(save_filename, (qr_array*255).astype(np.uint8))
    return np.array(qr_codes)  # Shape: (num_qr, size, size)

# ----------------------- Main Dataset Preparation -----------------------


if __name__ == "__main__":
    # 1. Augment Cover Images
    print("Augmenting cover images...")
    load_and_augment_images(dataset_path, augmented_path, target_size=(256, 256), num_augmented=10)
    
    # 2. Generate a pool of 100 QR Codes
    print("Generating QR code pool...")
    qr_texts = [f"Hidden Watermark {i+1}" for i in range(100)]
    qr_codes_pool = generate_qr_codes(qr_texts, watermark_path, size=256)
    
    # Save the QR code pool for later use (e.g., in encoder training)
    np.save(os.path.join(watermark_path, "qr_pool.npy"), qr_codes_pool)
    
    print("Dataset preparation complete.")
