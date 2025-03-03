import cv2
import pywt
import numpy as np
import qrcode
from pyzbar.pyzbar import decode
from reedsolo import RSCodec
from scipy.fftpack import dct, idct

# =================================
#   1. Helper / Utility Functions
# =================================

def add_reed_solomon(data_str, ecc_bytes=32):
    """
    Encode a string with Reed-Solomon for extra error correction.
    Returns bytes suitable for QR encoding.
    """
    rs = RSCodec(ecc_bytes)
    encoded = rs.encode(data_str.encode('utf-8'))
    return encoded  # bytes

def remove_reed_solomon(data_bytes, ecc_bytes=32):
    """
    Decode Reed-Solomon bytes back to the original string.
    """
    rs = RSCodec(ecc_bytes)
    decoded = rs.decode(data_bytes)
    return decoded.decode('utf-8')

def generate_qr(data_bytes, size=256):
    """
    Generate a high-ECC QR code from data_bytes, then resize to 'size'.
    By default, we use version=None so qrcode can choose automatically.
    """
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H
    )
    qr.add_data(data_bytes)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color='black', back_color='white').convert('L')
    # Resize to the desired dimension
    qr_img = qr_img.resize((size, size), resample=cv2.INTER_NEAREST)
    return np.array(qr_img, dtype=np.uint8)

def adaptive_threshold_morph(img_gray):
    """
    Apply adaptive thresholding and morphological closing to clean noise.
    """
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        img_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,  # Adjust for image size
        C=2
    )
    # Morphological closing to fill small holes
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return cleaned

# ---------------------
# SPREAD-SPECTRUM UTILS
# ---------------------

def spread_spectrum_embed(dct_subband, watermark_img, alpha=0.05, seed=12345):
    """
    Spread-spectrum embedding in DCT:
      - Generate a pseudo-random pattern (PN) for each pixel in the watermark.
      - If watermark pixel ~ 1 => add alpha * PN, else subtract alpha * PN.
    """
    h, w = watermark_img.shape
    # Ensure sub-band can hold the watermark
    if h > dct_subband.shape[0] or w > dct_subband.shape[1]:
        raise ValueError("Watermark is larger than sub-band region!")

    rng = np.random.default_rng(seed)
    # Uniform random in [-1,1]
    pn = rng.uniform(low=-1.0, high=1.0, size=(h, w))

    dct_embed = np.copy(dct_subband)
    for y in range(h):
        for x in range(w):
            bit = 1 if watermark_img[y, x] >= 128 else 0
            sign = +1 if bit == 1 else -1
            dct_embed[y, x] += alpha * sign * pn[y, x]

    return dct_embed

def spread_spectrum_extract(dct_subband, h, w, alpha=0.05, seed=12345):
    """
    Spread-spectrum extraction:
      - Generate the same PN sequence.
      - Dot product sign => bit=1 or bit=0.
    Returns a grayscale image of shape (h, w).
    """
    rng = np.random.default_rng(seed)
    pn = rng.uniform(low=-1.0, high=1.0, size=(h, w))

    extracted_img = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            val = dct_subband[y, x]
            dot = val * pn[y, x]
            if dot >= 0:
                extracted_img[y, x] = 255  # bit=1
            else:
                extracted_img[y, x] = 0    # bit=0
    return extracted_img

# ====================================
#   2. Watermark Embedding Function
# ====================================

def dwt_dct_embed_qr(
    image_path,
    secret_text,
    output_path,
    wavelet='db4',
    alpha=0.05,
    ecc_bytes=32,
    seed=12345
):
    """
    3-Level Wavelet (db4), multi-sub-band embedding (LH3, HL3, HH3) with spread-spectrum DCT.
    - Larger QR (256x256) for better extraction.
    - Additional Reed-Solomon ECC on top of QR ECC.
    - Final stego image is saved to output_path.
    """

    # 1) Read & convert to YCrCb
    cover = cv2.imread(image_path)
    if cover is None:
        raise ValueError(f"Could not open image: {image_path}")
    cover_ycrcb = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(cover_ycrcb)

    # 2) 3-Level DWT on Y channel
    coeffs = pywt.wavedec2(y, wavelet=wavelet, level=3)
    # structure: LL3, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1)
    LL3, (LH3, HL3, HH3), sub2, sub1 = coeffs

    # 3) Reed-Solomon encode + QR code
    rs_encoded = add_reed_solomon(secret_text, ecc_bytes=ecc_bytes)
    # We create a large QR code for better post-distortion readability
    qr_img = generate_qr(rs_encoded, size=256)

    # 4) For each sub-band, do DCT => Spread-Spectrum => iDCT
    def embed_subband(band):
        # forward DCT
        dct_band = dct(dct(band.T, norm='ortho').T, norm='ortho')
        # embed
        dct_band_embed = spread_spectrum_embed(dct_band, qr_img, alpha=alpha, seed=seed)
        # inverse DCT
        return idct(idct(dct_band_embed.T, norm='ortho').T, norm='ortho')

    LH3_e = embed_subband(LH3)
    HL3_e = embed_subband(HL3)
    HH3_e = embed_subband(HH3)

    # 5) Reconstruct Y channel
    new_coeffs = (LL3, (LH3_e, HL3_e, HH3_e), sub2, sub1)
    y_reconstructed = pywt.waverec2(new_coeffs, wavelet=wavelet)
    # shape correction
    y_reconstructed = cv2.resize(y_reconstructed, (y.shape[1], y.shape[0]))
    y_reconstructed = np.clip(y_reconstructed, 0, 255).astype(np.uint8)

    # 6) Merge & save stego
    merged_ycrcb = cv2.merge([y_reconstructed, cr, cb])
    stego_bgr = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite(output_path, stego_bgr)
    cv2.imwrite('embedded_qr.png', qr_img)

# ====================================
#   3. Watermark Extraction Function
# ====================================

def dwt_dct_extract_qr(
    image_path,
    wavelet='db4',
    alpha=0.05,
    ecc_bytes=32,
    seed=12345
):
    """
    Extract the QR code from the stego image:
    1) 3-Level DWT => sub-bands
    2) For LH3, HL3, HH3 => DCT => Spread-Spectrum extract => partial QR
    3) Median-fuse sub-band results => adaptive threshold => morphological ops
    4) Decode => Reed-Solomon => final text
    """

    stego = cv2.imread(image_path)
    if stego is None:
        raise ValueError(f"Could not open image: {image_path}")
    stego_ycrcb = cv2.cvtColor(stego, cv2.COLOR_BGR2YCrCb)
    y, _, _ = cv2.split(stego_ycrcb)

    # 1) 3-Level DWT
    coeffs = pywt.wavedec2(y, wavelet=wavelet, level=3)
    LL3, (LH3, HL3, HH3), sub2, sub1 = coeffs

    # 2) For each sub-band => DCT => spread-spectrum extraction
    def extract_subband(band):
        dct_band = dct(dct(band.T, norm='ortho').T, norm='ortho')
        # we know we used a 256x256 watermark
        # but ensure the sub-band is large enough
        h, w = band.shape
        wm_size = min(256, h, w)
        extracted = spread_spectrum_extract(dct_band, wm_size, wm_size, alpha=alpha, seed=seed)
        return extracted

    ext_LH3 = extract_subband(LH3)
    ext_HL3 = extract_subband(HL3)
    ext_HH3 = extract_subband(HH3)

    # 3) Combine sub-band results using median
    combined = np.median(
        np.stack([ext_LH3, ext_HL3, ext_HH3], axis=-1),
        axis=-1
    ).astype(np.uint8)

    # 4) Adaptive threshold + morphological ops
    cleaned = adaptive_threshold_morph(combined)

    # 5) Decode QR
    decoded_info = decode(cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR))
    if not decoded_info:
        return None

    # 6) Reed-Solomon decode
    raw_data = decoded_info[0].data
    try:
        text = remove_reed_solomon(raw_data, ecc_bytes=ecc_bytes)
        return text
    except:
        return None

# ====================================
#         DEMO / EXAMPLE
# ====================================
if __name__ == '__main__':
    cover_image = 'original.png'   # Path to your cover image
    stego_image = 'stego.png'      # Output stego image
    secret_text = 'Hello from multi-band DWT + DCT + Spread Spectrum!'
    
    # 1) Embed
    dwt_dct_embed_qr(
        image_path=cover_image,
        secret_text=secret_text,
        output_path=stego_image,
        wavelet='db4',   # or 'bior4.4'
        alpha=0.05,      # Adjust for your use case
        ecc_bytes=32,
        seed=12345
    )
    
    # 2) Extract
    extracted_msg = dwt_dct_extract_qr(
        image_path=stego_image,
        wavelet='db4',
        alpha=0.05,
        ecc_bytes=32,
        seed=12345
    )
    if extracted_msg:
        print("Extracted message:", extracted_msg)
    else:
        print("Failed to decode any QR code from the stego image.")
