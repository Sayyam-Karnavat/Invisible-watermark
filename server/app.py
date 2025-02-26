import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Configure folders for uploads and extracted images
UPLOAD_FOLDER = 'uploads'
EXTRACT_FOLDER = 'extracted'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Use the same secret key as used during embedding
SECRET_KEY = 'ssk'
ALPHA = 5

# --- Utility Functions for DCT/IDCT, Moiré Generation, and Extraction ---

def dct2(block):
    """Compute 2D Discrete Cosine Transform."""
    return cv2.dct(np.float32(block))

def idct2(block):
    """Compute Inverse 2D Discrete Cosine Transform."""
    return cv2.idct(block)

def generate_moire_pattern(rows, cols, secret_key):
    """Generate a moiré pattern using sinusoidal gratings."""
    np.random.seed(abs(hash(secret_key)) % (2**32))
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    X, Y = np.meshgrid(x, y)

    f1 = np.random.uniform(10, 20)  # Base frequency
    f2 = f1 + np.random.uniform(1, 5)  # Slightly different frequency
    angle1 = np.random.uniform(0, np.pi)
    angle2 = angle1 + np.random.uniform(0, np.pi / 18)

    grating1 = np.sin(2 * np.pi * f1 * (X * np.cos(angle1) + Y * np.sin(angle1)))
    grating2 = np.sin(2 * np.pi * f2 * (X * np.cos(angle2) + Y * np.sin(angle2)))

    moire = (grating1 - grating2) / 2.0  # Normalize
    return moire

def extract_moire_from_image(image, secret_key, alpha=ALPHA):
    """
    Extract the moiré pattern from the watermarked image using the secret key.
    
    It assumes that during embedding the moiré pattern was added into the blue
    channel's DCT coefficients in the mid-frequency region (rows//4:rows//4*3, cols//4:cols//4*3)
    scaled by the factor alpha.
    """
    # Convert the blue channel to float32 and compute its DCT
    blue_channel = np.float32(image[:, :, 0])
    dct_coeff = dct2(blue_channel)
    rows, cols = blue_channel.shape
    mid_x, mid_y = rows // 4, cols // 4

    # Extract the modified mid-frequency block (which contains the embedded moiré)
    extracted_block = dct_coeff[mid_x:mid_x*3, mid_y:mid_y*3] / alpha

    # (Optional) Regenerate the expected moiré pattern using the secret key
    # expected_moire = generate_moire_pattern(rows, cols, secret_key)
    # expected_region = expected_moire[mid_x:mid_x*3, mid_y:mid_y*3]
    # You could compare extracted_block vs. expected_region here if desired.

    # Place the extracted block back into a zero matrix and perform the inverse DCT
    extracted_dct = np.zeros_like(dct_coeff)
    extracted_dct[mid_x:mid_x*3, mid_y:mid_y*3] = extracted_block
    moire_extracted = idct2(extracted_dct)
    
    # Normalize the recovered moiré pattern for saving/display (0-255)
    moire_extracted_norm = cv2.normalize(moire_extracted, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(moire_extracted_norm)

# --- Flask Endpoints ---

@app.route("/")
def homepage():
    return render_template("upload.html")

@app.route('/verify', methods=['POST'])
def verify():
    """
    Expects an uploaded PNG image file with the key 'image'.
    Extracts the moiré pattern using the secret key and saves it locally.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    # Secure and save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Read the image using OpenCV (ensure proper reading of PNG images)
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Failed to read image.'}), 400

    try:
        # Extract the moiré pattern using the provided secret key
        extracted_moire = extract_moire_from_image(image, SECRET_KEY, alpha=ALPHA)
        # Save the extracted moiré pattern image
        extracted_filename = 'extracted_' + filename
        extracted_path = os.path.join(EXTRACT_FOLDER, extracted_filename)
        cv2.imwrite(extracted_path, extracted_moire)
    except Exception as e:
        return jsonify({'error': f'Extraction failed: {str(e)}'}), 500

    return jsonify({
        'message': 'Moiré pattern extracted successfully.',
        'extracted_image': extracted_filename
    }), 200

if __name__ == '__main__':
    # Run the server on all interfaces (0.0.0.0) at port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
