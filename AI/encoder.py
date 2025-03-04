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
# Utility Functions: 2D DCT and Inverse DCT
#####################################
def dct2d(x):
    """
    Applies a 2D Discrete Cosine Transform (DCT-II) along the height and width dimensions.
    Input shape: [batch, height, width, channels]
    Output shape: [batch, height, width, channels]
    """
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
    """
    Applies a 2D Inverse Discrete Cosine Transform (IDCT, corresponding to DCT-III)
    along the height and width dimensions.
    Input shape: [batch, height, width, channels]
    Output shape: [batch, height, width, channels]
    """
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
# Custom Layers for Safe Serialization
#####################################
@tf.keras.utils.register_keras_serializable()
class IdentityLayer(layers.Layer):
    def call(self, inputs):
        return inputs

    def get_config(self):
        config = super().get_config()
        return config

@tf.keras.utils.register_keras_serializable()
class DCT2DLayer(layers.Layer):
    def call(self, inputs):
        return dct2d(inputs)

    def get_config(self):
        config = super().get_config()
        return config

@tf.keras.utils.register_keras_serializable()
class IDCT2DLayer(layers.Layer):
    def call(self, inputs):
        return idct2d(inputs)

    def get_config(self):
        config = super().get_config()
        return config

#####################################
# Self-Attention Mechanism (Updated)
#####################################
@tf.keras.utils.register_keras_serializable()
class SelfAttention(layers.Layer):
    def __init__(self, filters=None, **kwargs):
        """
        If filters is None, the layer will use the input channels.
        """
        super(SelfAttention, self).__init__(**kwargs)
        self.filters = filters
        self.softmax = layers.Softmax(axis=-1)
        self.residual_conv = None

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self.in_channels = in_channels  # store as Python int
        if self.filters is None:
            self.filters = in_channels
        self.query_conv = layers.Conv2D(self.filters, kernel_size=1, padding="same")
        self.key_conv = layers.Conv2D(self.filters, kernel_size=1, padding="same")
        self.value_conv = layers.Conv2D(self.filters, kernel_size=1, padding="same")
        if self.filters != self.in_channels:
            self.residual_conv = layers.Conv2D(self.filters, kernel_size=1, padding="same")
        else:
            self.residual_conv = IdentityLayer()  # Use custom identity layer
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        batch = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        q = self.query_conv(inputs)
        k = self.key_conv(inputs)
        v = self.value_conv(inputs)
        
        # Reshape to (batch, height*width, self.filters)
        q_reshaped = tf.reshape(q, [batch, height * width, self.filters])
        k_reshaped = tf.reshape(k, [batch, height * width, self.filters])
        v_reshaped = tf.reshape(v, [batch, height * width, self.filters])
        
        attention_map = tf.matmul(q_reshaped, k_reshaped, transpose_b=True)  # (batch, H*W, H*W)
        attention_map = self.softmax(attention_map)
        attended_features = tf.matmul(attention_map, v_reshaped)  # (batch, H*W, self.filters)
        
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

    # --- Cover branch: extract features from cover image ---
    x = layers.Conv2D(64, (7,7), strides=2, padding="same")(cover_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = res_block(x, 64)
    cover_features = res_block(x, 128, stride=2)  # Expected shape: (batch, 32, 32, 128)

    # --- Watermark branch: process QR code watermark ---
    w = layers.Conv2D(64, (3,3), activation="relu", padding="same")(watermark_input)
    w = layers.MaxPooling2D((2,2))(w)
    w = layers.Conv2D(128, (3,3), activation="relu", padding="same")(w)
    w = layers.MaxPooling2D((2,2))(w)
    w = layers.Conv2D(256, (3,3), activation="relu", padding="same")(w)
    w = layers.MaxPooling2D((2,2))(w)  # Expected shape: (batch, 32, 32, 256)
    w = layers.Conv2D(128, (3,3), activation="relu", padding="same")(w)  # Now shape: (batch, 32, 32, 128)
    # Project watermark branch to 256 channels.
    w = layers.Conv2D(256, (1,1), activation="relu", padding="same")(w)  # Now shape: (batch, 32, 32, 256)

    # --- Fusion: fuse cover features (after self-attention) and watermark branch ---
    attention_layer = SelfAttention(256)(cover_features)  # Output shape: (batch, 32, 32, 256)
    merged = layers.Add()([attention_layer, w])  # Both now have 256 channels.
    merged = layers.Conv2D(256, (3,3), activation="relu", padding="same")(merged)

    # --- Frequency domain embedding using DCT ---
    # Convert merged features to frequency domain.
    freq = DCT2DLayer(name="dct_transform")(merged)
    # Embed watermark information in frequency domain via a small convolution.
    freq_embedded = layers.Conv2D(256, (3,3), activation="relu", padding="same")(freq)
    # Combine the original frequency coefficients with the embedded ones.
    freq_merged = layers.Add()([freq, freq_embedded])
    # Convert back to spatial domain.
    merged_reconstructed = IDCT2DLayer(name="idct_transform")(freq_merged)

    # --- Reconstruction: upsample to get the encoded image ---
    x = layers.Conv2DTranspose(128, (3,3), strides=2, padding="same", activation="relu")(merged_reconstructed)
    x = layers.Conv2DTranspose(64, (3,3), strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, (3,3), strides=2, padding="same", activation="relu")(x)
    encoded_image = layers.Conv2D(3, (3,3), activation="sigmoid", padding="same", name="encoded_image")(x)

    encoder = models.Model(inputs=[cover_input, watermark_input], outputs=encoded_image, name="Encoder")
    return encoder

#####################################
# Data Pipeline
#####################################
def load_and_preprocess_image(filepath):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0,1]
    return image

def add_watermark(image):
    return (image, watermark_tensor), image

if __name__ == "__main__":

    augmented_dir = "dataset/augmented_images"
    file_pattern = os.path.join(augmented_dir, "*.png")
    cover_ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    cover_ds = cover_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    qr_watermark_np = generate_qr_code(data="Hidden Watermark", size=256)
    np.save("original_watermark.npy" , qr_watermark_np)
    qr_watermark_np = np.expand_dims(qr_watermark_np, axis=-1)
    watermark_tensor = tf.convert_to_tensor(qr_watermark_np, dtype=tf.float32)

    train_ds = cover_ds.map(add_watermark, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(8).prefetch(tf.data.AUTOTUNE)

    #####################################
    # Training
    #####################################
    encoder = build_encoder()
    encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="mse")
    encoder.fit(train_ds, epochs=200)
    encoder.save("encoder_model.keras")

    encoded_images = encoder.predict(train_ds)
    np.save("encoded_images.npy", encoded_images)
    
    print("Encoder training complete. Model saved.")
