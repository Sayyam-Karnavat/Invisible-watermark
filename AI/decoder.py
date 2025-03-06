import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pickle

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
# Image Preprocessing Functions
#####################################
def preprocess_image(image):
    """Ensure images are resized to (256, 256, 3)."""
    image = tf.image.resize(image, (256, 256))
    return image

def preprocess_qr(qr_image):
    """Ensure QR codes are resized to (256, 256, 1)."""
    qr_image = tf.image.resize(qr_image, (256, 256))
    qr_image = tf.expand_dims(qr_image, axis=-1)  # Ensure shape (256,256,1)
    return qr_image

#####################################
# Build Decoder Model
#####################################
def build_decoder(encoded_shape=(256,256,3), watermark_shape=(256,256,1)):
    encoded_input = layers.Input(shape=encoded_shape, name="encoded_image")
    
    # Downsampling: Extract deep features from the encoded image.
    x = layers.Conv2D(64, (3,3), strides=1, padding="same", activation="relu")(encoded_input)
    x = layers.MaxPooling2D((2,2), strides=2, padding="same")(x)  # 128x128
    x = layers.Conv2D(128, (3,3), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2,2), strides=2, padding="same")(x)  # 64x64
    x = layers.Conv2D(256, (3,3), strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2,2), strides=2, padding="same")(x)  # 32x32
    
    # Residual Connection: Save feature map for later use
    skip_connection = x

    # Frequency domain processing using DCT & IDCT
    freq = DCTLayer()(x)
    freq_processed = layers.Conv2D(256, (3,3), activation="relu", padding="same")(freq)
    x = layers.Subtract()([x, freq_processed])
    x = IDCTLayer()(x)

    # Restore details via residual connection
    x = layers.Add()([x, skip_connection])  

    # Upsampling: Recover spatial resolution for watermark.
    x = layers.Conv2DTranspose(128, (3,3), strides=2, padding="same", activation="relu")(x)  # 64x64x128
    x = layers.Conv2DTranspose(64, (3,3), strides=2, padding="same", activation="relu")(x)   # 128x128x64
    x = layers.Conv2DTranspose(32, (3,3), strides=2, padding="same", activation="relu")(x)   # 256x256x32
    
    # Final Refinement Block
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = layers.Conv2D(16, (3,3), activation="relu", padding="same")(x)
    
    # Output layer: Reconstruct watermark (1 channel)
    decoded_watermark = layers.Conv2D(1, (3,3), activation="sigmoid", padding="same", name="decoded_watermark")(x)
    
    decoder = models.Model(inputs=encoded_input, outputs=decoded_watermark, name="Decoder")
    return decoder

#####################################
# Load Data from Pickled DataFrame and Prepare Dataset
#####################################
def load_pickled_data(filepath):
    """Loads the pickled dataframe and extracts embedded images and expected QR codes."""
    with open(filepath, 'rb') as f:
        df = pickle.load(f)
    
    # Convert each embedded image and QR code to a consistent format.
    # Assumes that the DataFrame stores numpy arrays for both columns.
    encoded_images = np.array([preprocess_image(img).numpy() for img in df["embedded_image"]])
    expected_qrs = np.array([preprocess_qr(qr).numpy() for qr in df["QR_code"]])
    
    return encoded_images, expected_qrs

#####################################
# Training the Decoder and Updating DataFrame
#####################################
if __name__ == "__main__":
    # Load pickled dataset with columns: "embedded_image", "QR_code"
    pickled_filepath = "dataset/encoded_images.pkl"  # Update if necessary
    encoded_images, watermarks = load_pickled_data(pickled_filepath)
    
    # Create TensorFlow dataset for training
    decoder_ds = tf.data.Dataset.from_tensor_slices((encoded_images, watermarks))
    decoder_ds = decoder_ds.batch(8).prefetch(tf.data.AUTOTUNE)
    
    # Build and compile the decoder
    decoder = build_decoder()
    decoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="mse")
    
    # Train the decoder
    decoder.fit(decoder_ds, epochs=400)
    
    # Save the trained decoder model
    decoder.save("decoder_model.keras")
    
    # Load the DataFrame to update it with extracted outputs
    with open(pickled_filepath, 'rb') as f:
        df = pickle.load(f)
    
    # For each embedded image, predict the extracted QR code and update the DataFrame
    extracted_outputs = []
    for emb_img in df["embedded_image"]:
        # Ensure emb_img has shape (256,256,3)
        emb_img = preprocess_image(emb_img).numpy()  # Convert tensor to numpy array
        emb_img = np.expand_dims(emb_img, axis=0)  # Add batch dimension
        extracted_qr = decoder.predict(emb_img)[0]   # Remove batch dimension
        extracted_outputs.append(extracted_qr)
    
    # Add a new column "extracted_output" to the DataFrame
    df["extracted_output"] = extracted_outputs
    
    # Save the updated DataFrame back as a pickle file
    df.to_pickle("dataset/updated_encoded_images.pkl")
    
    print("Decoder training complete. Model saved and DataFrame updated with extracted outputs.")
