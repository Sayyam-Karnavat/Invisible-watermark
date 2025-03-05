
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
# import tensorflow as tf


import pandas as pd


df = pd.read_pickle("dataset/updated_encoded_images.pkl")

print(df.head())