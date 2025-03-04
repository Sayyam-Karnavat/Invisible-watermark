import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import qrcode

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#####################################
# Custom Layers for DCT and IDCT (Serializable)
#####################################
class DCTLayer(layers.Layer):
    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 3, 1])  # [B, W, C, H]
        x = tf.signal.dct(x, type=2, norm='ortho', axis=-1)
        x = tf.transpose(x, perm=[0, 3, 1, 2])  # [B, H, W, C]

        x = tf.transpose(x, perm=[0, 1, 3, 2])  # [B, H, C, W]
        x = tf.signal.dct(x, type=2, norm='ortho', axis=-1)
        x = tf.transpose(x, perm=[0, 1, 3, 2])  # [B, H, W, C]
        return x

    def get_config(self):
        return super().get_config()

class IDCTLayer(layers.Layer):
    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 3, 1])  # [B, W, C, H]
        x = tf.signal.idct(x, type=3, norm='ortho', axis=-1)
        x = tf.transpose(x, perm=[0, 3, 1, 2])  # [B, H, W, C]

        x = tf.transpose(x, perm=[0, 1, 3, 2])  # [B, H, C, W]
        x = tf.signal.idct(x, type=3, norm='ortho', axis=-1)
        x = tf.transpose(x, perm=[0, 1, 3, 2])  # [B, H, W, C]
        return x

    def get_config(self):
        return super().get_config()

#####################################
# Utility: Generate QR Code Watermark
#####################################
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

#####################################
# Build Decoder Model
#####################################

def build_decoder(encoded_shape=(256,256,3), watermark_shape=(256,256,1)):
    encoded_input = layers.Input(shape=encoded_shape, name="encoded_image")
    
    # Downsampling: Extract deep features from the encoded image.
    x = layers.Conv2D(64, (3,3), strides=1, padding="same", activation="relu")(encoded_input)  # Keep stride=1 for finer details
    x = layers.MaxPooling2D((2,2), strides=2, padding="same")(x)  # Downsample to 128x128
    
    x = layers.Conv2D(128, (3,3), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2,2), strides=2, padding="same")(x)  # Downsample to 64x64
    
    x = layers.Conv2D(256, (3,3), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2,2), strides=2, padding="same")(x)  # Downsample to 32x32
    
    # Residual Connection: Save this feature map for later use
    skip_connection = x

    # Apply DCT and process frequency domain features.
    freq = DCTLayer()(x)
    freq_processed = layers.Conv2D(256, (3,3), activation="relu", padding="same")(freq)
    x = layers.Subtract()([x, freq_processed])
    x = IDCTLayer()(x)

    # Add Residual Skip Connection to restore lost details
    x = layers.Add()([x, skip_connection])  

    # Upsampling: Recover spatial resolution for the watermark.
    x = layers.Conv2DTranspose(128, (3,3), strides=2, padding="same", activation="relu")(x)  # 64x64x128
    x = layers.Conv2DTranspose(64, (3,3), strides=2, padding="same", activation="relu")(x)   # 128x128x64
    x = layers.Conv2DTranspose(32, (3,3), strides=2, padding="same", activation="relu")(x)   # 256x256x32
    
    # Final Refinement Block: Helps improve details before final output.
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = layers.Conv2D(16, (3,3), activation="relu", padding="same")(x)
    
    # Output layer: Reconstruct the watermark (1 channel).
    decoded_watermark = layers.Conv2D(1, (3,3), activation="sigmoid", padding="same", name="decoded_watermark")(x)
    
    decoder = models.Model(inputs=encoded_input, outputs=decoded_watermark, name="Decoder")
    return decoder





if __name__ == "__main__":
    

    #####################################
    # Prepare Decoder Training Data
    #####################################
    # Load the encoded images saved from the encoder.
    encoded_images = np.load("encoded_images.npy")  # shape: (num_examples, 256,256,3)

    # Generate the watermark that was used during encoding.
    qr_watermark_np = generate_qr_code(data="Hidden Watermark", size=256)
    qr_watermark_np = np.expand_dims(qr_watermark_np, axis=-1)  # shape: (256,256,1)

    # Since the same watermark was used for all examples, create labels by repeating the watermark.
    num_examples = encoded_images.shape[0]
    watermarks = np.repeat(np.expand_dims(qr_watermark_np, axis=0), num_examples, axis=0)

    # Create a dataset for decoder training.
    decoder_ds = tf.data.Dataset.from_tensor_slices((encoded_images, watermarks))
    decoder_ds = decoder_ds.batch(8).prefetch(tf.data.AUTOTUNE)

    #####################################
    # Training the Decoder
    #####################################
    decoder = build_decoder()
    decoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="mse")
    decoder.fit(decoder_ds, epochs=100)

    # Save the model with custom layers
    decoder.save("decoder_model.keras")

    print("Decoder training complete. Model saved.")
