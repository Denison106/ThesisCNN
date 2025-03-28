import numpy as np
import h5py
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import TensorBoard
import os

from amp_gen import generate_training_data
from network import build_model, prepare_callbacks
from utils import visualize_data, plot_learning_curves

# Experimental System Parameters
exp_sys_params = {
    "pix_size": 0.1997,  # Pixel size in micrometers
    "wavelength": 0.651,  # Wavelength in micrometers
    "ri_immersion": 1.518,  # Refractive index of immersion medium
    "beam_tilt_angle": 36.3 * (np.pi / 180),  # Beam tilt angle in radians
    "detector_size": (258, 258),  # Detector size in pixels (height, width)
}

# Display Parameters
display_params = {
    "display_pause_time": 1,  # Pause time for visualization
}

# Training Parameters
training_params = {
    "angles_number": 1,  # Number of azimuth angles per image
    "batch_size": 4,  # Batch size for training
    "epochs": 5,  # Number of epochs for training
    "learning_rate": 0.001,  # Learning rate for optimizer
    "validation_split": 0.2  # 20% of data for validation
}

# Define paths (Update these paths for your system)
dataset_path = [
    "/home/deniz/.cache/kagglehub/datasets/imsparsh/flowers-dataset/versions/2/train/daisy",
    "/home/deniz/.cache/kagglehub/datasets/imsparsh/flowers-dataset/versions/2/train/dandelion",
    "/home/deniz/.cache/kagglehub/datasets/imsparsh/flowers-dataset/versions/2/train/rose",
    "/home/deniz/.cache/kagglehub/datasets/imsparsh/flowers-dataset/versions/2/train/sunflower",
    "/home/deniz/.cache/kagglehub/datasets/imsparsh/flowers-dataset/versions/2/train/tulip"
]

save_path = "/home/deniz/DeepVID/Data/dataset.h5"  # Path to save the dataset

# TensorBoard Log Directory
log_dir = "logs/fit/" + tf.keras.callbacks.TensorBoard().log_dir
os.makedirs(log_dir, exist_ok=True)

def data_generator(dataset_file, indices, batch_size):
    """
    Generates batches of data to prevent RAM overload.

    Parameters:
        dataset_file (str): Path to the HDF5 dataset.
        indices (list): Indices for data selection.
        batch_size (int): The batch size for training.

    Yields:
        (batch_inputs, batch_targets): A batch of data for training.
    """
    while True:  # Infinite generator for training
        with h5py.File(dataset_file, "r") as hf:
            inputs = hf["input_images"]
            targets = hf["output_labels"]

            total_samples = len(indices)
            for start_idx in range(0, total_samples, batch_size):
                end_idx = min(start_idx + batch_size, total_samples)
                batch_inputs = inputs[start_idx:end_idx]
                batch_targets = targets[start_idx:end_idx]
                yield batch_inputs, batch_targets


def preprocessed_dataset(save_path, batch_size, validation_split):
    """
    Loads dataset in a memory-efficient way using TensorFlow `Dataset` API.

    Parameters:
        save_path (str): Path to the HDF5 dataset.
        batch_size (int): The batch size for training.
        validation_split (float): Percentage of data to use for validation.

    Returns:
        train_dataset, validation_dataset, test_dataset: TensorFlow dataset objects.
    """
    with h5py.File(save_path, "r") as hf:
        inputs = hf["input_images"]
        targets = hf["output_labels"]

        total_samples = inputs.shape[0]
        val_test_size = int(total_samples * validation_split)
        val_size = val_test_size // 2
        train_size = total_samples - val_test_size

        train_indices = range(0, train_size)
        val_indices = range(train_size, train_size + val_size)
        test_indices = range(train_size + val_size, total_samples)

        train_gen = lambda: data_generator(save_path, train_indices, batch_size)
        val_gen = lambda: data_generator(save_path, val_indices, batch_size)
        test_gen = lambda: data_generator(save_path, test_indices, batch_size)

        train_dataset = Dataset.from_generator(
            train_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, *inputs.shape[1:]), dtype=tf.float32),
                tf.TensorSpec(shape=(None, *targets.shape[1:]), dtype=tf.float32),
            )
        )

        validation_dataset = Dataset.from_generator(
            val_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, *inputs.shape[1:]), dtype=tf.float32),
                tf.TensorSpec(shape=(None, *targets.shape[1:]), dtype=tf.float32),
            )
        )

        test_dataset = Dataset.from_generator(
            test_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, *inputs.shape[1:]), dtype=tf.float32),
                tf.TensorSpec(shape=(None, *targets.shape[1:]), dtype=tf.float32),
            )
        )

        train_dataset = train_dataset.shuffle(1024).repeat().prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.repeat().prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    # Generate dataset if not already created
    if dataset_path and save_path:
        print("Generating dataset...")
        generate_training_data(exp_sys_params, training_params, dataset_path, save_path)

    # Load dataset in a memory-efficient way
    print("Loading dataset using preprocessed_dataset()...")
    train_dataset, validation_dataset, test_dataset = preprocessed_dataset(
        save_path, training_params["batch_size"], training_params["validation_split"]
    )

    # Build the CNN model
    model = build_model(input_shape=(258, 258, 1), learning_rate=training_params["learning_rate"])

    # Create TensorBoard Callback
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model using the dataset
    history = model.fit(
        train_dataset,
        epochs=training_params["epochs"],
        steps_per_epoch=100,  # Adjust this based on dataset size
        validation_data=validation_dataset,
        validation_steps=20,  # Adjust this based on dataset size
        callbacks=[tensorboard_callback]  # Include TensorBoard Callback
    )

    # Plot learning curves
    plot_learning_curves(history)

    print(f"Training completed. Run TensorBoard with: tensorboard --logdir={log_dir}")
