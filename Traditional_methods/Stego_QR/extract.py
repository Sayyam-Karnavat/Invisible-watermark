import cv2
import numpy as np
import qrcode
from PIL import Image
from pyzbar.pyzbar import decode

def extract_qr(stego_img_path, region=None):
    """
    Extract and decode the QR code from the stego image by reading the same region in which
    it was embedded. If no region is provided, the region is computed dynamically as the central
    half of the image.
    
    Parameters:
      stego_img_path (str): Path to the stego image.
      region (tuple): Optional. (x_offset, y_offset, width, height) defining the extraction region.
    
    Returns:
      decoded_text (str or None): The text decoded from the QR code.
      extracted_img (PIL.Image): The binary QR image extracted from the stego image.
    """
    # Open stego image and ensure it's in RGB mode.
    stego_img = Image.open(stego_img_path).convert("RGB")
    stego_width, stego_height = stego_img.size

    # Compute dynamic region if not provided.
    if region is None:
        x_offset = stego_width // 4
        y_offset = stego_height // 4
        region_width = stego_width // 2
        region_height = stego_height // 2
        region = (x_offset, y_offset, region_width, region_height)
    else:
        x_offset, y_offset, region_width, region_height = region

    # Reconstruct the embedded QR code by reading the LSB of the red channel.
    extracted_img = Image.new("L", (region_width, region_height))
    stego_pixels = stego_img.load()
    for x in range(region_width):
        for y in range(region_height):
            r, g, b = stego_pixels[x_offset + x, y_offset + y]
            bit = r & 1
            pixel_value = 255 if bit == 1 else 0
            extracted_img.putpixel((x, y), pixel_value)

    # Optionally, save the extracted QR image.
    extracted_img.save("extracted_qr.png")

    # Use pyzbar to decode the QR code.
    decoded_objects = decode(extracted_img)
    decoded_text = decoded_objects[0].data.decode("utf-8") if decoded_objects else None

    return decoded_text, extracted_img

if __name__ == '__main__':
    cover_img_path = "original.png"     # Provide your cover image path.
    stego_img_path = "stego.png"     # Output stego image path.
    

    # Extract and decode the QR code watermark.
    decoded_text, extracted_img = extract_qr(stego_img_path)
    if decoded_text:
        print("Decoded QR text from watermark:", decoded_text)
    else:
        print("Failed to decode the QR watermark.")
