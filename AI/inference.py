import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim



def mse(imageA, imageB):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((imageA - imageB) ** 2)


def psnr(imageA, imageB):
    """ Compute Peak Signal-to-Noise Ratio (PSNR) """
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')  # Perfect match
    return 10 * np.log10(1.0 / mse_value)  # Assuming images are normalized [0,1]

def normalized_cross_correlation(imageA, imageB):
    """ Compute Normalized Cross-Correlation (NCC) """
    imageA = imageA.flatten()
    imageB = imageB.flatten()
    return np.corrcoef(imageA, imageB)[0, 1]



# Load images
original_image = cv2.imread("dataset/original.png")
encoded_image = np.load("encoded_image.npy")[0]

extracted_watermark = np.load("extracted_watermark.npy").squeeze()  # Remove batch dim
original_watermark = np.load("original_watermark.npy").squeeze()  # Ensure correct shape

# Convert images
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
encoded_image = (encoded_image * 255).astype(np.uint8)
extracted_watermark = (extracted_watermark * 255).astype(np.uint8)
original_watermark = (original_watermark * 255).astype(np.uint8)


# Compute similarity metrics
mse_value = mse(original_watermark, extracted_watermark)
psnr_value = psnr(original_watermark, extracted_watermark)
ssim_value = ssim(original_watermark, extracted_watermark, data_range=1.0)
ncc_value = normalized_cross_correlation(original_watermark, extracted_watermark)

# Print results
print(f"MSE: {mse_value:.5f}")
print(f"PSNR: {psnr_value:.2f} dB")
print(f"SSIM: {ssim_value:.5f}")
print(f"NCC: {ncc_value:.5f}")

# Plot images
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(original_image); ax[0].set_title("Original Image")
ax[1].imshow(encoded_image); ax[1].set_title("Watermark Embedded Image")
ax[2].imshow(original_watermark, cmap="gray"); ax[2].set_title("Original Watermark")
ax[3].imshow(extracted_watermark, cmap="gray"); ax[3].set_title("Extracted Watermark")

# Remove axis labels
for a in ax:
    a.axis("off")

plt.show()
