import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import test_model, load_sino_and_azim
from network import build_winnik_model, scaled_sigmoid_pi

#sino_path = r"C:\Users\jw\Desktop\dyplomy\Erkosar Deniz\test_data\sim_sino_beads.mat"
sino_path = "/mnt/c/Users/deniz/Desktop/Denison/Uni/THESIS/test_data/test_data/exp_sino.mat"
weights_path = "/home/deniz/DeepVID/trained_model/epoch_20_model_checkpoint.keras"

sino, azim_vec = load_sino_and_azim(sino_path)

model = build_winnik_model(input_shape=(256, 256, 1), learning_rate=0.0001)

try:
    model.load_weights(weights_path)
    print("Weights loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model weights: {e}")

display = True
indices = list(range(0, 180, 30))
if azim_vec is not None:
    azim_vec = azim_vec[indices]
sino = sino[:, :, indices]

pred_output = test_model(model, sino, display, gt_azimuth=azim_vec)
