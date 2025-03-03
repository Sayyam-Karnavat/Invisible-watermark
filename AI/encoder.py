import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
import qrcode
from PIL import Image

# Set random seed for reproducibility
np.random.seed(42)

#####################################
# Utility: Generate QR Code Watermark
#####################################
def generate_qr_code(data="Hidden Watermark", size=256):
    """
    Generates a QR code with high error correction,
    converts it to grayscale, resizes it to (size, size),
    and normalizes pixel values to [0,1].
    """
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
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

#####################################
# Utility: Residual Block (Custom ResNet)
#####################################
def res_block(x, filters, kernel_size=3, stride=1):
    """A simple residual block with skip connection."""
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

#####################################
# Build the Encoder Model
#####################################
def build_encoder(image_shape=(256,256,3), watermark_shape=(256,256,1)):
    """
    Builds a custom encoder that embeds a QR code watermark into the cover image.
    The cover branch uses a custom ResNet–style feature extractor while the
    watermark branch downsamples the QR code to a matching resolution.
    """
    # Define inputs
    cover_input = layers.Input(shape=image_shape, name="cover_image")
    watermark_input = layers.Input(shape=watermark_shape, name="watermark")
    
    # --------------- Cover Branch (Custom ResNet) ---------------
    # Initial convolution: from 256x256x3 -> 128x128x64
    x = layers.Conv2D(64, (7,7), strides=2, padding="same")(cover_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Further downsampling: 128x128 -> 64x64
    x = layers.MaxPooling2D((2,2))(x)
    # Residual block: keep 64 channels
    x = res_block(x, 64)
    # Downsample using a residual block with stride=2: output 32x32, 128 channels
    cover_features = res_block(x, 128, stride=2)
    
    # --------------- Watermark Branch (QR Code) ---------------
    # Input watermark: 256x256x1 (QR code)
    w = layers.Conv2D(64, (3,3), activation="relu", padding="same")(watermark_input)   # -> 256x256x64
    w = layers.MaxPooling2D((2,2))(w)                                                   # -> 128x128x64
    w = layers.Conv2D(128, (3,3), activation="relu", padding="same")(w)                 # -> 128x128x128
    w = layers.MaxPooling2D((2,2))(w)                                                   # -> 64x64x128
    w = layers.Conv2D(256, (3,3), activation="relu", padding="same")(w)                 # -> 64x64x256
    w = layers.MaxPooling2D((2,2))(w)                                                   # -> 32x32x256
    w = layers.Conv2D(128, (3,3), activation="relu", padding="same")(w)                 # -> 32x32x128

    # --------------- Fusion ---------------
    # Both branches now have feature maps of shape 32x32 with 128 channels each.
    merged = layers.Concatenate()([cover_features, w])   # -> 32x32x256
    merged = layers.Conv2D(256, (3,3), activation="relu", padding="same")(merged)
    
    # --------------- Reconstruction (Upsampling) ---------------
    x = layers.Conv2DTranspose(128, (3,3), strides=2, padding="same", activation="relu")(merged)  # -> 64x64x128
    x = layers.Conv2DTranspose(64, (3,3), strides=2, padding="same", activation="relu")(x)          # -> 128x128x64
    x = layers.Conv2DTranspose(32, (3,3), strides=2, padding="same", activation="relu")(x)          # -> 256x256x32
    encoded_image = layers.Conv2D(3, (3,3), activation="sigmoid", padding="same", name="encoded_image")(x)

    encoder = models.Model(inputs=[cover_input, watermark_input], outputs=encoded_image, name="Encoder")
    return encoder

#####################################
# Data Pipeline for Encoder Training
#####################################
def load_and_preprocess_image(filepath):
    """
    Loads an image from disk (already 256x256, RGB), decodes it,
    sets the shape explicitly, and normalizes pixel values to [0,1].
    """
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape([256, 256, 3])
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Directory containing your already-prepared (augmented) cover images
augmented_dir = "dataset/augmented_images"
file_pattern = os.path.join(augmented_dir, "*.png")  # adjust extension as needed
cover_ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
cover_ds = cover_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Generate the single QR watermark and convert to tensor
qr_watermark_np = generate_qr_code(data="Hidden Watermark", size=256)  # shape: (256,256)
qr_watermark_np = np.expand_dims(qr_watermark_np, axis=-1)               # shape: (256,256,1)
watermark_tensor = tf.convert_to_tensor(qr_watermark_np, dtype=tf.float32)

# Function to pair each cover image with the same watermark and use the cover as target.
def add_watermark(image):
    return (image, watermark_tensor), image  # ([cover, watermark], target=cover)

train_ds = cover_ds.map(add_watermark, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(8).prefetch(tf.data.AUTOTUNE)

#####################################
# Build, Compile & Train Encoder
#####################################
encoder = build_encoder()
encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
encoder.summary()

# Train the encoder on the augmented images
encoder.fit(train_ds, epochs=100)
encoder.save("encoder_model.keras")

# (Optional) Generate encoded images on the training set for inspection
encoded_images = encoder.predict(train_ds)
np.save("encoded_images.npy", encoded_images)
print("Encoder training complete. Model saved.")
