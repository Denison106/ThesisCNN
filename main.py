import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import os

from amp_gen import generate_training_data
from network import build_model, prepare_callbacks
from utils import preprocessed_dataset, plot_learning_curves, visualize_data


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
    "epochs": 2,  # Number of epochs for training
    "learning_rate": 0.001,  # Learning rate for optimizer
    "validation_split": 0.2,  # 20% of data for validation
    "dataset size": 4000 #Total size of dataset (training+val+test)
}

# Define paths (Update these paths for your system)
dataset_path = [r"C:\Users\jw\Desktop\dyplomy\flowers_dataset"]

save_path = r"C:\Users\jw\Desktop\dyplomy\Erkosar Deniz\data\dataset.h5"  # Path to save the dataset

# TensorBoard Log Directory
log_dir = "logs/fit/" + tf.keras.callbacks.TensorBoard().log_dir
os.makedirs(log_dir, exist_ok=True)


if __name__ == "__main__":
    # Generate dataset if not already created
    if dataset_path and save_path:
        print("Generating dataset...")
        generate_training_data(exp_sys_params, training_params, dataset_path, save_path, training_params["dataset size"])

    # Load dataset in a memory-efficient way
    print("Loading dataset using preprocessed_dataset()...")
    train_dataset, validation_dataset, test_dataset = preprocessed_dataset(
        save_path, training_params["batch_size"], training_params["validation_split"]
    )

    # Build the CNN model
    model = build_model(input_shape=exp_sys_params["detector_size"]+(1,),
                        learning_rate=training_params["learning_rate"])

    # Create TensorBoard Callback
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint = ModelCheckpoint(
        os.path.join(os.getcwd(), r'trained_model/epoch_{epoch:02d}_model_checkpoint.keras'),
        save_freq="epoch"
    )

    # Train the model using the dataset
    train_data_size = (1.0 - training_params["validation_split"]) * training_params["dataset size"]
    steps_per_epoch = np.ceil(train_data_size / training_params["batch_size"]).astype(int)
    val_data_size = (training_params["validation_split"]/2) * training_params["dataset size"]
    val_steps_per_epoch = np.ceil(val_data_size / training_params["batch_size"]).astype(int)
    test_steps_per_epoch = val_steps_per_epoch

    model.summary()

    history = model.fit(
        train_dataset,
        epochs=training_params["epochs"],
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=val_steps_per_epoch,
        callbacks=[tensorboard_callback],  # Include TensorBoard Callback
        verbose=1
    )

    score = model.evaluate(test_dataset,
                           batch_size=training_params["batch_size"],
                           steps=test_steps_per_epoch,
                           verbose=1)

    print(f'Test loss: {score}')

    # Plot learning curves
    plot_learning_curves(history)

    print(f"Training completed. Run TensorBoard with: tensorboard --logdir={log_dir}")
