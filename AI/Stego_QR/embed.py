import qrcode
from PIL import Image
from pyzbar.pyzbar import decode

def embed_invisible_qr(cover_img_path, stego_img_path, text_to_encode, region=None):
    """
    Generate a QR code from the provided text and embed it invisibly into the cover image.
    The QR is embedded in the middle of the image; if no region is provided, the region is
    computed dynamically as the central half of the image.
    
    Parameters:
      cover_img_path (str): Path to the cover image.
      stego_img_path (str): Path where the stego image will be saved.
      text_to_encode (str): Text to encode in the QR code.
      region (tuple): Optional. (x_offset, y_offset, width, height) defining the embedding region.
    
    Returns:
      region (tuple): The region used for embedding.
    """
    # Open cover image and ensure it's in RGB mode.
    cover_img = Image.open(cover_img_path).convert("RGB")
    cover_width, cover_height = cover_img.size

    # Compute dynamic region (middle half) if not provided.
    if region is None:
        x_offset = cover_width // 4
        y_offset = cover_height // 4
        region_width = cover_width // 2
        region_height = cover_height // 2
        region = (x_offset, y_offset, region_width, region_height)
    else:
        x_offset, y_offset, region_width, region_height = region

    # Generate QR code.
    qr = qrcode.QRCode(
        version=1,  # low version; will be resized to fill the region
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(text_to_encode)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white").convert("L")

    # Resize the QR code to exactly match the embedding region dimensions.
    img_qr = img_qr.resize((region_width, region_height), Image.NEAREST)

    # Embed the QR code into the cover image by modifying the LSB of the red channel.
    cover_pixels = cover_img.load()
    for x in range(region_width):
        for y in range(region_height):
            qr_pixel = img_qr.getpixel((x, y))
            # Decide bit: 0 for black, 1 for white.
            bit = 0 if qr_pixel < 128 else 1
            r, g, b = cover_pixels[x_offset + x, y_offset + y]
            new_r = (r & ~1) | bit
            cover_pixels[x_offset + x, y_offset + y] = (new_r, g, b)

    cover_img.save(stego_img_path)
    print(f"Invisible QR embedded at region {region} in '{stego_img_path}'.")
    return region

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
    cover_img_path = "dataset/original.png"     # Provide your cover image path.
    stego_img_path = "dataset/stego.png"     # Output stego image path.
    text_to_encode = "This is a hidden QR code."

    # Embed the QR code invisibly into the middle of the cover image.
    region_used = embed_invisible_qr(cover_img_path, stego_img_path, text_to_encode)

    # Extract the QR code from the same region and decode its content.
    decoded_text, extracted_img = extract_qr(stego_img_path, region_used)
    if decoded_text:
        print("Decoded QR text:", decoded_text)
    else:
        print("QR code could not be decoded.")
