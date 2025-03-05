import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from encoder import SelfAttention, IdentityLayer, DCT2DLayer , IDCT2DLayer
from decoder import DCTLayer, IDCTLayer


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



decoder_model = tf.keras.models.load_model("decoder_model.keras",custom_objects=custom_objects_decoder, safe_mode=False)


original_image = cv2.imread("dataset/original_images/quality.png")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)


original_image = cv2.resize(original_image , (256 , 256))
original_image = original_image / 255.0


image_for_prediction = np.expand_dims(original_image , axis= 0)


predicted_image = decoder_model.predict(image_for_prediction)
predicted_image = np.squeeze(predicted_image)
predicted_image = (predicted_image * 255).astype(np.uint8)


# Show images using Matplotlib
plt.figure(figsize=(12, 4))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")

# Encoded Image
plt.subplot(1, 3, 2)
plt.imshow(predicted_image)
plt.title("Extracted QR ")
plt.axis("off")

plt.show()


# cv2.imshow("Without watermark original",image)
# cv2.waitKey(0)
# cv2.DestroyAllWindow()