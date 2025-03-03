import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import cv2
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def build_robust_encoder(image_shape=(256, 256, 3), watermark_shape=(256, 256, 1)):
    """
    Builds a robust encoder that embeds a watermark (of the same size as the image)
    into the cover image using a pretrained ResNet50 backbone for feature extraction.
    """
    # Define inputs
    image_input = layers.Input(shape=image_shape, name="cover_image")
    watermark_input = layers.Input(shape=watermark_shape, name="watermark")
    
    # -------------------- Cover Image Branch --------------------
    # Use ResNet50 pretrained on ImageNet (without the top) as feature extractor.
    # Note: We pass image_input directly to ResNet50.
    base_model = ResNet50(include_top=False, weights="imagenet", input_tensor=image_input)
    # Freeze the backbone for initial training
    for layer in base_model.layers:
        layer.trainable = False
    # Get deep image features (output shape e.g., (8,8,2048) for 256x256 input)
    image_features = base_model.output

    # -------------------- Watermark Branch --------------------
    # Process watermark (which is the same size as the image: 256x256x1)
    # We downsample it gradually to match the spatial resolution of image_features.
    w = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(watermark_input)  # (256,256,64)
    w = layers.MaxPooling2D(pool_size=(2, 2))(w)  # (128,128,64)
    w = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(w)  # (128,128,128)
    w = layers.MaxPooling2D(pool_size=(2, 2))(w)  # (64,64,128)
    w = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(w)  # (64,64,256)
    w = layers.MaxPooling2D(pool_size=(2, 2))(w)  # (32,32,256)
    w = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(w)  # (32,32,512)
    w = layers.MaxPooling2D(pool_size=(4, 4))(w)  # (8,8,512)

    # -------------------- Fusion --------------------
    # Fuse the image and watermark features at the same spatial resolution (8x8)
    merged = layers.Concatenate()([image_features, w])  # (8,8,2048+512)
    merged = layers.Conv2D(1024, (3, 3), activation="relu", padding="same")(merged)

    # -------------------- Reconstruction --------------------
    # Use transposed convolutions (upsampling) to reconstruct the encoded image
    x = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same", activation="relu")(merged)  # 16x16
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)       # 32x32
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)       # 64x64
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)        # 128x128
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)        # 256x256

    # Final reconstruction to output a 3-channel image
    encoded_image = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same", name="encoded_image")(x)

    encoder = models.Model(inputs=[image_input, watermark_input], outputs=encoded_image, name="Robust_Encoder")
    return encoder

# Build and compile the encoder with our new architecture.
encoder = build_robust_encoder(image_shape=(256, 256, 3), watermark_shape=(256, 256, 1))
encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

# -------------------- Data Utilities --------------------
def load_image(image_path , size = (256 , 256)):
    # Load image in RGB and do not resize (assumes image is already 256x256 or will be cropped externally)
    img = cv2.imread(image_path)
    img = cv2.resize(img , size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img.astype(np.float32)

def generate_and_save_watermark(size=(256, 256), save_path="original_watermark.npy"):
    """
    Generates a structured watermark using random circles and patterns, which is 
    easier to extract from an image after blending.
    
    - Uses a combination of noise, circles, and a Moire-like pattern.
    - Saves the watermark as a NumPy array.
    
    Returns:
        watermark (np.array): The generated watermark.
    """
    # Create an empty grayscale image (0-255)
    watermark = np.zeros(size, dtype=np.uint8)

    # Add random dots
    num_dots = 500  # Adjust for density
    for _ in range(num_dots):
        x, y = np.random.randint(0, size[1]), np.random.randint(0, size[0])
        cv2.circle(watermark, (x, y), radius=1, color=255, thickness=-1)

    # Add concentric circles
    num_circles = 10  # Number of circles
    center_x, center_y = size[1] // 2, size[0] // 2
    max_radius = min(size) // 3  # Largest circle size

    for i in range(num_circles):
        radius = int((i + 1) * (max_radius / num_circles))
        cv2.circle(watermark, (center_x, center_y), radius=radius, color=128, thickness=1)

    # Add Moire pattern (diagonal stripes)
    for i in range(0, size[0], 10):
        cv2.line(watermark, (0, i), (size[1], i), color=200, thickness=1)

    # Normalize to [0,1] range and save
    watermark = watermark.astype(np.float32) / 255.0
    np.save(save_path, watermark)
    return watermark

def generate_grid_lattice(size=(256, 256), save_path="original_watermark.npy"):
    """
    Generates a watermark with a structured grid lattice pattern.
    """
    watermark = np.zeros(size, dtype=np.uint8)
    spacing = 20  # Adjust for density
    
    for x in range(0, size[1], spacing):
        for y in range(0, size[0], spacing):
            cv2.circle(watermark, (x, y), radius=2, color=255, thickness=-1)
    
    watermark = watermark.astype(np.float32) / 255.0
    np.save(save_path, watermark)
    return watermark

# -------------------- Training --------------------
# Load your training image (assume it's 256x256; if larger, crop or pre-resize as needed)
image_path = "dataset/original.png"
cover_image = load_image(image_path)  # shape: (256,256,3)

# Generate and save watermark (same size as cover image)
watermark = generate_and_save_watermark(size=(cover_image.shape[0], cover_image.shape[1]))

# Expand dimensions for batch training
cover_image = np.expand_dims(cover_image, axis=0)  # (1,256,256,3)
watermark = np.expand_dims(watermark, axis=0)        # (1,256,256,1)

# Train the encoder
encoder.fit([cover_image, watermark], cover_image, epochs=1000, batch_size=1)

# Save the trained model and the encoded image for later use
encoder.save("encoder_model.keras")
encoded_image = encoder.predict([cover_image, watermark])
np.save("encoded_image.npy", encoded_image)
print("Encoder training complete. Model saved.")