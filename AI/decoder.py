import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, Concatenate
import numpy as np
import cv2

np.random.seed(42)

def build_decoder():
    """
    Builds a U-Net–style decoder to extract the embedded watermark from the encoded image.
    The network takes a 256x256x3 encoded image and outputs a 256x256x1 watermark.
    """
    encoded_input = Input(shape=(256, 256, 3))
    
    # -------------------- Downsampling Path --------------------
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(encoded_input)  # 256x256
    p1 = MaxPooling2D((2, 2))(c1)  # 128x128
    
    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    p2 = MaxPooling2D((2, 2))(c2)  # 64x64
    
    c3 = Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    p3 = MaxPooling2D((2, 2))(c3)  # 32x32
    
    # -------------------- Bottleneck --------------------
    b = Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
    
    # -------------------- Upsampling Path (using Transposed Convolutions) --------------------
    u1 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(b)  # 64x64
    u1 = Concatenate()([u1, c3])
    
    u2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(u1)  # 128x128
    u2 = Concatenate()([u2, c2])
    
    u3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(u2)  # 256x256
    u3 = Concatenate()([u3, c1])
    
    # -------------------- Output Layer --------------------
    output = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(u3)
    
    decoder = models.Model(inputs=encoded_input, outputs=output, name="Decoder")
    return decoder

# Example loss: Combine SSIM loss with MSE (tuning loss weights as needed)
# def ssim_loss(y_true, y_pred):
#     return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Build and compile the decoder
decoder = build_decoder()
decoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=["mse"],
                loss_weights=[0.3])

# -------------------- Training Data --------------------
# Load the encoded image (should be of shape (1,256,256,3))
encoded_image = np.load("encoded_image.npy")
# Load the original watermark (shape: (256,256,1)) and add batch dimension
watermark = np.load("original_watermark.npy")
watermark = np.expand_dims(watermark, axis=0)  # Now shape: (1,256,256,1)

# (Optional) You can add distortions here if desired:
# distorted_encoded_image = np.array([apply_distortions(img) for img in encoded_image])
# For now, we'll train on the clean encoded image.
training_input = encoded_image


# -------------------- Train the Decoder --------------------
decoder.fit(training_input, watermark, epochs=1000, batch_size=1)

# Save the extracted watermark and the trained decoder
extracted_watermark = decoder.predict(training_input)
np.save("extracted_watermark.npy", extracted_watermark)
decoder.save("decoder_model.keras")
print("Decoder training complete. Model saved.")