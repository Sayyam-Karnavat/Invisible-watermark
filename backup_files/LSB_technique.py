import cv2
import numpy as np

def embed_watermark(image_path, secret_key, output_path):
    """
    Embed a pseudo-random binary watermark into the blue channel's LSB of an image.
    
    Parameters:
      image_path (str): Path to the input image.
      secret_key (str): Secret key to seed the watermark generation.
      output_path (str): Path to save the watermarked image.
    """
    # Load image (BGR format)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    
    rows, cols, _ = img.shape
    
    # Seed the random generator using a hash of the secret key for reproducibility
    np.random.seed(abs(hash(secret_key)) % (2**32))
    
    # Generate a binary watermark: a matrix of 0s and 1s with dimensions matching the image's height and width
    watermark = np.random.randint(0, 2, (rows, cols), dtype=np.uint8)
    
    # Create a copy of the image to embed the watermark
    watermarked_img = img.copy()
    
    # Embed watermark in the blue channel (index 0 in BGR)
    # Clear the LSB of the blue channel then OR with the watermark bit
    watermarked_img[:, :, 0] = (watermarked_img[:, :, 0] & 0xFE) | watermark
    
    # Save the watermarked image
    cv2.imwrite(output_path, watermarked_img)
    print("Watermark embedded and image saved to", output_path)

def extract_watermark(image_path, secret_key):
    """
    Extract the watermark from the watermarked image using the secret key.
    
    Parameters:
      image_path (str): Path to the watermarked image.
      secret_key (str): Secret key used during embedding.
    
    Returns:
      extracted_watermark (np.ndarray): The extracted binary watermark.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    
    rows, cols, _ = img.shape
    
    # Extract the watermark from the blue channel's LSB
    extracted_watermark = img[:, :, 0] & 1
    
    # (Optional) Regenerate the expected watermark pattern for verification
    np.random.seed(abs(hash(secret_key)) % (2**32))
    expected_watermark = np.random.randint(0, 2, (rows, cols), dtype=np.uint8)
    
    # For demonstration, print if the extraction matches the expected watermark
    if np.array_equal(extracted_watermark, expected_watermark):
        print("Watermark extraction successful and verified!")
    else:
        print("Warning: Extracted watermark does not match expected pattern.")
    
    return extracted_watermark

if __name__ == '__main__':
    # Example usage:
    input_image = 'product_image.jpg'         # Replace with your packaging image
    watermarked_image = 'watermarked.png'
    secret_key = 'sanyam_karnavat'     # This key should be kept secure
    
    # Embed the watermark
    embed_watermark(input_image, secret_key, watermarked_image)
    
    # Extract the watermark for verification
    watermark = extract_watermark(watermarked_image, secret_key)
    
    # Save the extracted watermark as an image (scaled to 0 or 255 for visibility)
    cv2.imwrite('extracted_watermark.png', watermark * 255)
