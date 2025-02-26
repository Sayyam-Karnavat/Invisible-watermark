import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# ------------------------
# Utility functions
# ------------------------

def load_image(image_path):
    """Load an image in grayscale mode and return as a numpy uint8 array."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return np.array(img, dtype=np.uint8)

def show_image(img, title="Image"):
    """Display an image using matplotlib."""
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# ------------------------
# QIM Helper Function
# ------------------------

def embed_qim(v, bit, delta):
    """
    Embed a single bit into a value v using QIM.
    
    Parameters:
        v (float): The value to be quantized (typically the real part of an FFT coefficient).
        bit (int): The watermark bit (0 or 1) to embed.
        delta (float): The quantization step.
        
    Returns:
        float: The quantized value encoding the bit.
    """
    Q = np.round(v / delta)
    # Adjust Q so that its parity matches the bit (even for 0, odd for 1)
    if (Q % 2) != bit:
        # Choose the adjustment (up or down) that brings v closer to Q*delta.
        Q_down = Q - 1
        Q_up = Q + 1
        err_down = abs(v - Q_down * delta)
        err_up = abs(v - Q_up * delta)
        Q = Q_down if err_down < err_up else Q_up
    return Q * delta

# ------------------------
# FFTT Watermark Embedding using QIM
# ------------------------

def fftt_embed(image, watermark, delta=10):
    """
    Embed an invisible watermark into a grayscale image using FFT and QIM.
    
    The watermark text is converted into a binary string. Then, in the FFT domain,
    we select a square region centered around the DC component. For each coefficient
    in that region (until all bits are embedded), we quantize its real part so that the
    quantization index’s parity encodes the watermark bit.
    
    Parameters:
        image (np.array): Input grayscale image (0-255).
        watermark (str): The watermark text to embed.
        delta (float): The quantization step size (controls embedding strength).
        
    Returns:
        np.array: The watermarked image (uint8).
    """
    # Convert watermark text to binary (8 bits per character)
    binary_watermark = ''.join(format(ord(char), '08b') for char in watermark)
    total_bits = len(binary_watermark)
    
    # Convert image to float32 for processing and compute its FFT
    image = image.astype(np.float32)
    fft_image = np.fft.fft2(image)
    rows, cols = image.shape
    
    # Choose a square region of size K x K such that K*K >= total_bits.
    K = math.ceil(math.sqrt(total_bits))
    start_x = rows // 2 - K // 2
    start_y = cols // 2 - K // 2
    bit_idx = 0
    
    # Embed watermark bits into the selected region
    for i in range(K):
        for j in range(K):
            if bit_idx < total_bits:
                x = start_x + i
                y = start_y + j
                bit = int(binary_watermark[bit_idx])
                coeff = fft_image[x, y]
                # Only modify the real part using QIM; leave the imaginary part intact.
                new_real = embed_qim(coeff.real, bit, delta)
                fft_image[x, y] = new_real + 1j * coeff.imag
                bit_idx += 1
            else:
                break

    # Apply inverse FFT to reconstruct the spatial image
    watermarked_img = np.fft.ifft2(fft_image).real
    watermarked_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)
    return watermarked_img

# ------------------------
# FFTT Watermark Extraction using QIM
# ------------------------

def fftt_extract(watermarked_img, watermark_length, delta=10):
    """
    Extract an invisible watermark from a grayscale image.
    
    The extraction applies FFT to the watermarked image, then reads the same square region
    used during embedding. For each coefficient, it computes the quantization index (by
    rounding the real part divided by delta) and extracts the bit as the parity of that index.
    
    Parameters:
        watermarked_img (np.array): The watermarked grayscale image.
        watermark_length (int): The length of the watermark text (in characters).
        delta (float): The quantization step size (must match the embedding parameter).
        
    Returns:
        str: The extracted watermark text.
    """
    total_bits = watermark_length * 8
    fft_watermarked = np.fft.fft2(watermarked_img)
    rows, cols = watermarked_img.shape
    
    K = math.ceil(math.sqrt(total_bits))
    start_x = rows // 2 - K // 2
    start_y = cols // 2 - K // 2
    extracted_bits = []
    bit_idx = 0
    
    for i in range(K):
        for j in range(K):
            if bit_idx < total_bits:
                x = start_x + i
                y = start_y + j
                coeff = fft_watermarked[x, y]
                # Recover the quantization index from the real part and take its parity.
                Q = np.round(coeff.real / delta)
                bit = int(Q % 2)
                extracted_bits.append(bit)
                bit_idx += 1
            else:
                break

    # Convert the bit sequence to text (8 bits per character)
    extracted_watermark = ""
    for i in range(0, total_bits, 8):
        byte_bits = extracted_bits[i:i+8]
        byte_val = int("".join(map(str, byte_bits)), 2)
        extracted_watermark += chr(byte_val)
    return extracted_watermark

# ------------------------
# Testing the Implementation
# ------------------------

if __name__ == "__main__":
    image_path = "images/penguin.png"  # Replace with your image path
    original_img = load_image(image_path)

    watermark_text = "FFTT_SECURE"
    # Use a delta (quantization step) that is strong enough; you may need to tune this value.
    delta = 20

    # Embed the watermark into the image
    watermarked_img = fftt_embed(original_img, watermark_text, delta=delta)
    cv2.imwrite("results/watermarked_fftt.png", watermarked_img)
    # Optionally display the watermarked image:
    # show_image(watermarked_img, "Watermarked Image")

    # Load the watermarked image (simulate a blind extraction)
    watermarked_img_loaded = load_image("results/watermarked_fftt.png")
    extracted_watermark = fftt_extract(watermarked_img_loaded, len(watermark_text), delta=delta)
    print("Extracted Watermark:", extracted_watermark)
