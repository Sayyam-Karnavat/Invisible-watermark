import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import qrcode
from PIL import Image

# Set TensorFlow environment flag to optimize performance
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

# Define paths
dataset_path = "dataset/original_images"   # Folder containing original images
watermark_path = "dataset/qr_watermarks"   # Folder to save generated QR codes
augmented_path = "dataset/augmented_images"  # Folder to save augmented images

# Create necessary directories
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

# ----------------------- Load and Augment Cover Images -----------------------

def load_and_augment_images(dataset_path, save_path, target_size=(256, 256), num_augmented=100):
    """
    Loads images from dataset_path, applies augmentation, and saves them to save_path.
    """
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(dataset_path, filename)
            
            # Load and resize image
            image = cv2.imread(image_path)
            image = cv2.resize(image, target_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB format
            
            # Expand dimensions to match ImageDataGenerator input shape
            image = np.expand_dims(image, axis=0)  # Shape: (1, 256, 256, 3)
            
            # Generate augmented images
            aug_iter = datagen.flow(image, batch_size=1)
            for i in range(num_augmented):
                augmented_image = next(aug_iter)[0]  # Get the first batch
                
                # Convert to uint8 and save
                augmented_image = (augmented_image * 255).astype(np.uint8)
                save_filename = os.path.join(save_path, f"{os.path.splitext(filename)[0]}_aug{i}.png")
                cv2.imwrite(save_filename, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

# ----------------------- Generate QR Code Watermarks -----------------------

def generate_qr_codes(data_list, save_path, size=256):
    """
    Generates QR code images from a list of text data and saves them as grayscale images.
    """
    for i, data in enumerate(data_list):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # Convert to grayscale image and resize
        img = qr.make_image(fill_color="black", back_color="white").convert("L")
        img = img.resize((size, size))
        
        # Convert to NumPy array and normalize
        qr_array = np.array(img, dtype=np.uint8)
        save_filename = os.path.join(save_path, f"qr_{i}.png")
        cv2.imwrite(save_filename, qr_array)

# ----------------------- Run Dataset Preparation -----------------------

# 1. Augment Cover Images
load_and_augment_images(dataset_path, augmented_path)

# # 2. Generate QR Codes for Watermarking
# sample_watermark_texts = ["Hidden Watermark 1", "Secret Key", "QR Code 12345"]
# generate_qr_codes(sample_watermark_texts, watermark_path)

# print("Dataset preparation complete.")
