import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import qrcode
from PIL import Image

# Set random seed for reproducibility
np.random.seed(42)

#####################################
# Build the Decoder Model
#####################################
def build_decoder(input_shape=(256,256,3)):
    """
    Builds a decoder network that extracts the QR code watermark from the encoded image.
    The network downscales the encoded image and then upsamples to recover the watermark.
    """
    encoded_input = layers.Input(shape=input_shape, name="encoded_input")
    
    # ------------------ Downsampling Path ------------------
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(encoded_input)
    x = layers.MaxPooling2D((2,2))(x)  # -> 128x128x64
    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)  # -> 64x64x128
    x = layers.Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)  # -> 32x32x256
    x = layers.Conv2D(512, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)  # -> 16x16x512

    # ------------------ Bottleneck ------------------
    x = layers.Conv2D(512, (3,3), activation="relu", padding="same")(x)
    
    # ------------------ Upsampling Path ------------------
    x = layers.Conv2DTranspose(512, (3,3), strides=2, padding="same", activation="relu")(x)  # -> 32x32x512
    x = layers.Conv2DTranspose(256, (3,3), strides=2, padding="same", activation="relu")(x)  # -> 64x64x256
    x = layers.Conv2DTranspose(128, (3,3), strides=2, padding="same", activation="relu")(x)  # -> 128x128x128
    x = layers.Conv2DTranspose(64, (3,3), strides=2, padding="same", activation="relu")(x)   # -> 256x256x64
    extracted_watermark = layers.Conv2D(1, (3,3), activation="sigmoid", padding="same", name="extracted_watermark")(x)

    decoder = models.Model(inputs=encoded_input, outputs=extracted_watermark, name="Decoder")
    return decoder

#####################################
# Data Pipeline for Decoder Training
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

# Directory containing your augmented cover images (same as used for encoder training)
augmented_dir = "dataset/augmented_images"
file_pattern = os.path.join(augmented_dir, "*.png")
cover_ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
cover_ds = cover_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
cover_ds = cover_ds.batch(8)

# Load the saved encoder model to generate encoded images on-the-fly
encoder = tf.keras.models.load_model("encoder_model.keras")

# Generate the same QR watermark as ground truth
def generate_qr_code(data="Hidden Watermark", size=256):
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

qr_watermark_np = generate_qr_code(data="Hidden Watermark", size=256)
qr_watermark_np = np.expand_dims(qr_watermark_np, axis=-1)  # (256,256,1)
watermark_tensor = tf.convert_to_tensor(qr_watermark_np, dtype=tf.float32)

# For each batch of cover images, generate encoded images and the corresponding watermark label
def generate_encoded_and_label(cover_batch):
    batch_size = tf.shape(cover_batch)[0]
    # Tile the same watermark for the batch
    watermark_batch = tf.tile(tf.expand_dims(watermark_tensor, axis=0), [batch_size,1,1,1])
    encoded = encoder([cover_batch, watermark_batch], training=False)
    return encoded, watermark_batch

train_ds_decoder = cover_ds.map(generate_encoded_and_label, num_parallel_calls=tf.data.AUTOTUNE)
train_ds_decoder = train_ds_decoder.prefetch(tf.data.AUTOTUNE)

#####################################
# Build, Compile & Train Decoder
#####################################
decoder = build_decoder()
decoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

# Train the decoder on the encoded images generated from augmented cover images
decoder.fit(train_ds_decoder, epochs=100)
decoder.save("decoder_model.keras")

# (Optional) Save one example of extracted watermark for inspection
for encoded_batch, watermark_batch in train_ds_decoder.take(1):
    extracted = decoder.predict(encoded_batch)
    np.save("extracted_watermark.npy", extracted)
    np.save("original_watermark.npy", watermark_batch.numpy())
    break

print("Decoder training complete. Model saved.")
