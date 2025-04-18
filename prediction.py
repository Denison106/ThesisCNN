import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from utils import test_model, load_sino_and_azim
from network import scaled_sigmoid_pi


#sino_path = r"C:\Users\jw\Desktop\dyplomy\Erkosar Deniz\test_data\sim_sino_beads.mat"
sino_path = r"C:\Users\jw\Desktop\dyplomy\Erkosar Deniz\test_data\exp_sino.mat"
model_path =\
    r"C:\Users\jw\Desktop\dyplomy\Erkosar Deniz\Deniz code\ThesisCNN\trained_model\model_checkpoint.keras"

sino, azim_vec = load_sino_and_azim(sino_path)

try:
    model = tf.keras.models.load_model(model_path, custom_objects={"scaled_sigmoid_pi": scaled_sigmoid_pi})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

display = True

# select only few projections from sinogram
indices = list(range(0, 180, 30))
if azim_vec:
    azim_vec = azim_vec[indices]
sino = sino[:, :, indices]

pred_output = test_model(model, sino, display, gt_azimuth=azim_vec)