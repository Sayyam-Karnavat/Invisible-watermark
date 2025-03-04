import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
# Disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
import tensorflow as tf

# Import custom layers from encoder and decoder
from encoder import SelfAttention, IdentityLayer, DCT2DLayer , IDCT2DLayer
from decoder import DCTLayer, IDCTLayer



def mse(imageA, imageB):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((imageA - imageB) ** 2)

def psnr(imageA, imageB):
    """ Compute Peak Signal-to-Noise Ratio (PSNR) """
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')  # Perfect match
    return 10 * np.log10(255.0**2 / mse_value)  # Assuming pixel range [0,255]

def normalized_cross_correlation(imageA, imageB):
    """ Compute Normalized Cross-Correlation (NCC) """
    imageA = imageA.flatten()
    imageB = imageB.flatten()
    return np.corrcoef(imageA, imageB)[0, 1]

# Define the custom objects dictionary for model deserialization
custom_objects_encoder = {
    'SelfAttention': SelfAttention,
    'IdentityLayer': IdentityLayer,
    'DCT2DLayer': DCT2DLayer,
    'IDCT2DLayer' : IDCT2DLayer
}


custom_objects_decoder = {
    "DCTLayer" : DCTLayer,
    "IDCTLayer" :IDCTLayer
}

# Load models with custom objects
encoder = tf.keras.models.load_model("encoder_model.keras", custom_objects=custom_objects_encoder, safe_mode=False)
decoder = tf.keras.models.load_model("decoder_model.keras", custom_objects=custom_objects_decoder, safe_mode=False)

# Load original image and watermark
original_image = cv2.imread("dataset/original_images/original_aug1.png")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Load original QR watermark
original_watermark = np.load("original_watermark.npy")[0]

# Resize the original image to match encoder input size (256, 256, 3)
original_image = cv2.resize(original_image, (256, 256))
# Resize the watermark to match the encoder's watermark input size (256, 256)
original_watermark = cv2.resize(original_watermark, (256, 256))
# Convert watermark to grayscale if needed
if len(original_watermark.shape) == 3:
    original_watermark = cv2.cvtColor(original_watermark, cv2.COLOR_BGR2GRAY)

# Normalize images (scale pixel values to [0, 1])
original_image = original_image / 255.0
original_watermark = original_watermark / 255.0

# Expand dimensions to create a batch
original_image_batch = np.expand_dims(original_image, axis=0)  # (1, 256, 256, 3)
original_watermark_batch = np.expand_dims(original_watermark, axis=(0, -1))  # (1, 256, 256, 1)

# Use the encoder to embed the watermark into the image
encoded_image_batch = encoder.predict([original_image_batch, original_watermark_batch])
encoded_image = np.squeeze(encoded_image_batch)  # Remove batch dimension
encoded_image = (encoded_image * 255).astype(np.uint8)  # Convert back to uint8 for visualization

# Use the decoder to extract the watermark from the encoded image
extracted_watermark_batch = decoder.predict(np.expand_dims(encoded_image / 255.0, axis=0))
extracted_watermark = np.squeeze(extracted_watermark_batch)  # Remove batch dimension
extracted_watermark = (extracted_watermark * 255).astype(np.uint8)  # Convert back to uint8

# Compute similarity metrics
mse_value = mse(original_watermark, extracted_watermark)
psnr_value = psnr(original_watermark, extracted_watermark)
ssim_value = ssim(original_watermark, extracted_watermark, data_range=255)
ncc_value = normalized_cross_correlation(original_watermark, extracted_watermark)

# Print results
print(f"MSE: {mse_value:.5f}")
print(f"PSNR: {psnr_value:.2f} dB")
print(f"SSIM: {ssim_value:.5f}")
print(f"NCC: {ncc_value:.5f}")

# Visualize the results
plt.figure(figsize=(12, 4))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original Image (Resized)")
plt.axis("off")

# Encoded Image with QR Watermark
plt.subplot(1, 3, 2)
plt.imshow(encoded_image)
plt.title("Encoded Image with QR Watermark")
plt.axis("off")

# Extracted QR Watermark
plt.subplot(1, 3, 3)
plt.imshow(extracted_watermark, cmap="gray")
plt.title("Extracted QR Watermark")
plt.axis("off")

plt.show()
