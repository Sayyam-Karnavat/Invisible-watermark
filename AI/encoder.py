import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import qrcode
import random
import pandas as pd


# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#####################################
# Utility Functions: 2D DCT and Inverse DCT
#####################################
def dct2d(x):
    # Apply DCT on height axis:
    x = tf.transpose(x, perm=[0, 2, 3, 1])  # [B, W, C, H]
    x = tf.signal.dct(x, type=2, norm='ortho', axis=-1)
    x = tf.transpose(x, perm=[0, 3, 1, 2])  # [B, H, W, C]
    # Apply DCT on width axis:
    x = tf.transpose(x, perm=[0, 1, 3, 2])  # [B, H, C, W]
    x = tf.signal.dct(x, type=2, norm='ortho', axis=-1)
    x = tf.transpose(x, perm=[0, 1, 3, 2])  # [B, H, W, C]
    return x

def idct2d(x):
    # Apply inverse DCT on height axis:
    x = tf.transpose(x, perm=[0, 2, 3, 1])  # [B, W, C, H]
    x = tf.signal.idct(x, type=3, norm='ortho', axis=-1)
    x = tf.transpose(x, perm=[0, 3, 1, 2])  # [B, H, W, C]
    # Apply inverse DCT on width axis:
    x = tf.transpose(x, perm=[0, 1, 3, 2])  # [B, H, C, W]
    x = tf.signal.idct(x, type=3, norm='ortho', axis=-1)
    x = tf.transpose(x, perm=[0, 1, 3, 2])  # [B, H, W, C]
    return x



#####################################
# Custom Layers for Safe Serialization
#####################################
@tf.keras.utils.register_keras_serializable()
class IdentityLayer(layers.Layer):
    def call(self, inputs):
        return inputs
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class DCT2DLayer(layers.Layer):
    def call(self, inputs):
        return dct2d(inputs)
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class IDCT2DLayer(layers.Layer):
    def call(self, inputs):
        return idct2d(inputs)
    def get_config(self):
        return super().get_config()

#####################################
# Self-Attention Mechanism (Updated)
#####################################
@tf.keras.utils.register_keras_serializable()
class SelfAttention(layers.Layer):
    def __init__(self, filters=None, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.filters = filters
        self.softmax = layers.Softmax(axis=-1)
        self.residual_conv = None
    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        if self.filters is None:
            self.filters = in_channels
        self.query_conv = layers.Conv2D(self.filters, kernel_size=1, padding="same")
        self.key_conv = layers.Conv2D(self.filters, kernel_size=1, padding="same")
        self.value_conv = layers.Conv2D(self.filters, kernel_size=1, padding="same")
        if self.filters != in_channels:
            self.residual_conv = layers.Conv2D(self.filters, kernel_size=1, padding="same")
        else:
            self.residual_conv = IdentityLayer()
        super(SelfAttention, self).build(input_shape)
    def call(self, inputs, **kwargs):
        batch = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        q = self.query_conv(inputs)
        k = self.key_conv(inputs)
        v = self.value_conv(inputs)
        q_reshaped = tf.reshape(q, [batch, height * width, self.filters])
        k_reshaped = tf.reshape(k, [batch, height * width, self.filters])
        v_reshaped = tf.reshape(v, [batch, height * width, self.filters])
        attention_map = tf.matmul(q_reshaped, k_reshaped, transpose_b=True)
        attention_map = self.softmax(attention_map)
        attended_features = tf.matmul(attention_map, v_reshaped)
        attended_features = tf.reshape(attended_features, [batch, height, width, self.filters])
        residual = self.residual_conv(inputs)
        return attended_features + residual
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

#####################################
# Residual Block
#####################################
def res_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = SelfAttention(filters)(x)
    x = layers.Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU()(x)
    return x

#####################################
# Encoder Model with DCT Embedding
#####################################
def build_encoder(image_shape=(256,256,3), watermark_shape=(256,256,1)):
    cover_input = layers.Input(shape=image_shape, name="cover_image")
    watermark_input = layers.Input(shape=watermark_shape, name="watermark")
    # --- Cover branch ---
    x = layers.Conv2D(64, (7,7), strides=2, padding="same")(cover_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = res_block(x, 64)
    cover_features = res_block(x, 128, stride=2)  # (batch, 32, 32, 128)
    # --- Watermark branch ---
    w = layers.Conv2D(64, (3,3), activation="relu", padding="same")(watermark_input)
    w = layers.MaxPooling2D((2,2))(w)
    w = layers.Conv2D(128, (3,3), activation="relu", padding="same")(w)
    w = layers.MaxPooling2D((2,2))(w)
    w = layers.Conv2D(256, (3,3), activation="relu", padding="same")(w)
    w = layers.MaxPooling2D((2,2))(w)  # (batch, 32, 32, 256)
    w = layers.Conv2D(128, (3,3), activation="relu", padding="same")(w)  # (batch, 32, 32, 128)
    w = layers.Conv2D(256, (1,1), activation="relu", padding="same")(w)  # (batch, 32, 32, 256)
    # --- Fusion ---
    attention_layer = SelfAttention(256)(cover_features)
    merged = layers.Add()([attention_layer, w])
    merged = layers.Conv2D(256, (3,3), activation="relu", padding="same")(merged)
    # --- Frequency domain embedding ---
    freq = DCT2DLayer(name="dct_transform")(merged)
    freq_embedded = layers.Conv2D(256, (3,3), activation="relu", padding="same")(freq)
    freq_merged = layers.Add()([freq, freq_embedded])
    merged_reconstructed = IDCT2DLayer(name="idct_transform")(freq_merged)
    # --- Reconstruction ---
    x = layers.Conv2DTranspose(128, (3,3), strides=2, padding="same", activation="relu")(merged_reconstructed)
    x = layers.Conv2DTranspose(64, (3,3), strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, (3,3), strides=2, padding="same", activation="relu")(x)
    encoded_image = layers.Conv2D(3, (3,3), activation="sigmoid", padding="same", name="encoded_image")(x)
    encoder = models.Model(inputs=[cover_input, watermark_input], outputs=encoded_image, name="Encoder")
    return encoder

#####################################
# Data Pipeline and Negative Samples Setup
#####################################
def load_and_preprocess_image(filepath):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0,1]
    image = tf.image.resize(image, (256, 256))
    return image

def add_watermark(image, image_path, qr_codes_pool):
    """ Embed watermark 80% of the time, else use empty watermark """
    if random.random() < 0.8:  # 80% chance to embed a watermark
        qr_index = random.randint(0, len(qr_codes_pool) - 1)
        qr_code = qr_codes_pool[qr_index]
        is_embedded = 1  # QR is embedded
    else:  # 20% chance to use an empty watermark
        qr_code = np.zeros_like(qr_codes_pool[0])  # Create empty watermark
        qr_index = -1  # Special marker for empty watermark
        is_embedded = 0  # No QR embedded
    
    watermark = np.expand_dims(qr_code, axis=-1)
    return (image, tf.convert_to_tensor(watermark, dtype=tf.float32)), image, image_path, qr_index, is_embedded

if __name__ == "__main__":
    # Load augmented cover images
    augmented_dir = "dataset/augmented_images"
    file_pattern = os.path.join(augmented_dir, "*.png")
    cover_files = tf.io.gfile.glob(file_pattern)

    qr_codes_pool = np.load(os.path.join("dataset/qr_watermarks", "qr_pool.npy"))

    df_records = []  
    cover_images = []
    watermarks = []
    image_paths = []
    qr_indices = []
    is_qr_embedded_list = []  # New list for embedding status

    for file in cover_files:
        img = load_and_preprocess_image(file)
        (img_tensor, qr_tensor), _, _, qr_idx, is_embedded = add_watermark(img, file, qr_codes_pool)

        cover_images.append(img_tensor)  # TF tensor for cover image
        watermarks.append(qr_tensor)  # TF tensor for QR code image
        df_records.append((img_tensor.numpy(), qr_tensor.numpy(), None, qr_idx, is_embedded))  # Store QR index & embedding flag

    # Create TensorFlow dataset
    train_ds = tf.data.Dataset.from_tensor_slices(((cover_images, watermarks), cover_images))
    train_ds = train_ds.batch(5).prefetch(tf.data.AUTOTUNE)

    # Train the encoder
    encoder = build_encoder()
    encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="mse")
    encoder.fit(train_ds, epochs=500)
    encoder.save("encoder_model.keras")

    # Predict embedded images
    encoded_images = encoder.predict(train_ds)
    
    # Fill the third column in the DataFrame
    for i, enc_img in enumerate(encoded_images):
        df_records[i] = (df_records[i][0], df_records[i][1], enc_img, df_records[i][3], df_records[i][4])  # Keep QR index & flag

    # Create and save DataFrame
    df = pd.DataFrame(df_records, columns=["original_image", "QR_code", "embedded_image", "QR_index", "is_qr_embedded"])
    df.to_pickle("dataset/encoded_images.pkl")

    print("Encoder training complete. Model and dataset saved.")