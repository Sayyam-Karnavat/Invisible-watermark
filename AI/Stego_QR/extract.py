import cv2
import numpy as np
import qrcode
from PIL import Image
from pyzbar.pyzbar import decode

def generate_qr_binary(text, wm_size):
    """
    Generate a QR code from text, then resize it to (wm_size x wm_size) and
    convert it to a binary image (with values 0 or 1).
    """
    qr = qrcode.QRCode(
        version=1,  # version can be adjusted as needed
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white").convert("L")
    img_qr = np.array(img_qr)
    # Resize to a small binary image that will serve as the watermark (e.g., 32x32 bits)
    img_qr_resized = cv2.resize(img_qr, (wm_size, wm_size), interpolation=cv2.INTER_NEAREST)
    # Convert to binary: 0 for black and 1 for white
    _, binary_wm = cv2.threshold(img_qr_resized, 128, 1, cv2.THRESH_BINARY)
    return binary_wm

def embed_qr_watermark_dct(cover_img_path, stego_img_path, text_to_encode, block_size=8, wm_size=32, alpha=20):
    """
    Embed a QR code watermark robustly into the cover image using DCT-based watermarking.
    
    The QR code is generated from text_to_encode, resized to (wm_size x wm_size) bits, and
    each watermark bit is embedded into an (block_size x block_size) block in the center region
    of the cover image.
    
    Parameters:
      cover_img_path (str): Path to a grayscale cover image.
      stego_img_path (str): Path where the watermarked image will be saved.
      text_to_encode (str): The text to encode into the QR code.
      block_size (int): Size of each block for DCT (e.g., 8).
      wm_size (int): Number of watermark bits per side (the watermark will be wm_size x wm_size).
      alpha (float): Embedding strength.
    """
    # Load the cover image in grayscale.
    cover = cv2.imread(cover_img_path, cv2.IMREAD_GRAYSCALE)
    if cover is None:
        raise ValueError("Cover image not found.")
    cover = np.float32(cover)
    H, W = cover.shape

    # Determine the embedding region size in pixels.
    region_width = wm_size * block_size
    region_height = wm_size * block_size

    # Compute the top-left corner of the centered region.
    x_offset = (W - region_width) // 2
    y_offset = (H - region_height) // 2

    # Generate the binary QR watermark.
    wm = generate_qr_binary(text_to_encode, wm_size)
    # wm is an array of shape (wm_size, wm_size) with values 0 or 1.

    stego = cover.copy()

    # Loop over each watermark bit/block.
    for i in range(wm_size):
        for j in range(wm_size):
            # Determine block coordinates in the cover image.
            x_start = x_offset + j * block_size
            y_start = y_offset + i * block_size
            block = stego[y_start:y_start+block_size, x_start:x_start+block_size]
            # Compute the DCT of the block.
            dct_block = cv2.dct(block)
            # Modify a mid-frequency coefficient (e.g. at position [4,3]) based on the watermark bit.
            if wm[i, j] == 1:
                dct_block[4, 3] += alpha
            else:
                dct_block[4, 3] -= alpha
            # Compute the inverse DCT to reconstruct the block.
            block_idct = cv2.idct(dct_block)
            stego[y_start:y_start+block_size, x_start:x_start+block_size] = block_idct

    stego = np.clip(stego, 0, 255).astype(np.uint8)
    cv2.imwrite(stego_img_path, stego)
    print("Watermark embedded and stego image saved as", stego_img_path)

def extract_qr_watermark_dct(stego_img_path, block_size=8, wm_size=32, threshold=0):
    """
    Extract the QR code watermark from the stego image using DCT-based extraction.
    
    The extraction is performed over the centered region (of size wm_size*block_size).
    Each block's chosen DCT coefficient is examined to decide the embedded bit.
    
    Parameters:
      stego_img_path (str): Path to the stego image.
      block_size (int): Block size used during embedding.
      wm_size (int): Watermark dimensions (wm_size x wm_size).
      threshold (float): Decision threshold for determining embedded bits.
    
    Returns:
      decoded_text (str or None): The decoded text from the QR code (if successfully read).
      extracted_img (numpy array): The recovered binary QR watermark image.
    """
    stego = cv2.imread(stego_img_path, cv2.IMREAD_GRAYSCALE)
    if stego is None:
        raise ValueError("Stego image not found.")
    stego = np.float32(stego)
    H, W = stego.shape

    region_width = wm_size * block_size
    region_height = wm_size * block_size
    x_offset = (W - region_width) // 2
    y_offset = (H - region_height) // 2

    extracted_wm = np.zeros((wm_size, wm_size), dtype=np.uint8)

    for i in range(wm_size):
        for j in range(wm_size):
            x_start = x_offset + j * block_size
            y_start = y_offset + i * block_size
            block = stego[y_start:y_start+block_size, x_start:x_start+block_size]
            dct_block = cv2.dct(block)
            coeff = dct_block[4, 3]
            extracted_wm[i, j] = 1 if coeff > threshold else 0

    # Convert the extracted watermark to a viewable image (scaling 0/1 to 0/255).
    extracted_img = (extracted_wm * 255).astype(np.uint8)
    cv2.imwrite("extracted_qr.png", extracted_img)
    print("Extracted watermark saved as extracted_qr.png")

    # Attempt to decode the extracted watermark as a QR code using pyzbar.
    pil_img = Image.fromarray(extracted_img)
    decoded_objs = decode(pil_img)
    decoded_text = decoded_objs[0].data.decode("utf-8") if decoded_objs else None
    return decoded_text, extracted_img

if __name__ == '__main__':
    cover_img_path = "dataset/original.png"     # Provide your cover image path.
    stego_img_path = "dataset/stego.png"     # Output stego image path.
    text_to_encode = "This is a hidden QR code."

    # Embed the QR code watermark robustly.
    embed_qr_watermark_dct(cover_img_path, stego_img_path, text_to_encode,
                             block_size=8, wm_size=32, alpha=20)

    # Extract and decode the QR code watermark.
    decoded_text, extracted_img = extract_qr_watermark_dct(stego_img_path,
                                                            block_size=8, wm_size=32,
                                                            threshold=0)
    if decoded_text:
        print("Decoded QR text from watermark:", decoded_text)
    else:
        print("Failed to decode the QR watermark.")
