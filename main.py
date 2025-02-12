import numpy as np
from data_gen import generate_training_data
from network import build_model
from utils import visualize_data, plot_learning_curves
import h5py

# Experimental System Parameters
exp_sys_params = {
    "pix_size": 3.5,  # Pixel size in micrometers
    "optical_mag": 100,  # Optical magnification
    "wavelength": 0.5,  # Wavelength in micrometers
    "ri_immersion": 1.33,  # Refractive index of immersion medium
    "beam_tilt_angle": 40 * (np.pi / 180),  # Beam tilt angle in radians
    "detector_size": (512, 512),  # Detector size in pixels (height, width)
}

# Display Parameters
display_params = {
    "display_pause_time": 1,  # Pause time for visualization
}

# Training Parameters
training_params = {
    "angles_number": 10,  # Number of azimuth angles per image
    "batch_size": 4,  # Batch size for training
    "epochs": 5,  # Number of epochs for training
    "learning_rate": 0.001,  # Learning rate for optimizer
}

# Define paths (FILL THESE IN)
dataset_path = "/Users/deniz/Desktop/Deniz/Uni/THESIS/DeepVID/Flowers Recognition_files"  # Path to folder containing phase object images
save_path = "/Users/deniz/Desktop/Deniz/Uni/THESIS/DeepVID/Data"  # Path to save the dataset

if __name__ == "__main__":
    # Generate dataset (if not already created)
    if dataset_path and save_path:
        print("Generating dataset...")
        generate_training_data(exp_sys_params, training_params, dataset_path, save_path)

    # Load dataset from HDF5 file
    print("Loading dataset from HDF5 file...")
    with h5py.File(save_path, "r") as hf:
        input_images = np.array(hf["input_images"])
        output_labels = np.array(hf["output_labels"])

    # Visualize some sample images
    visualize_data(input_images, display_params)

    # Build and train the model
    model = build_model(input_shape=input_images.shape[1:], learning_rate=training_params["learning_rate"])
    history = model.fit(input_images, output_labels, batch_size=training_params["batch_size"], epochs=training_params["epochs"])

    # Plot learning curves
    plot_learning_curves(history)
