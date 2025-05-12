
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import os
from tensorflow.keras.layers import Lambda

from network import norm_vec
from data_generator import generate_training_data
from utils import preprocessed_dataset, plot_learning_curves


# Experimental System Parameters
exp_sys_params = {
    "pix_size": 0.1997,  # Pixel size in micrometers
    "wavelength": 0.651,  # Wavelength in micrometers
    "ri_immersion": 1.518,  # Refractive index of immersion medium
    "beam_tilt_angle": 36.3 * (np.pi / 180),  # Beam tilt angle in radians
    "detector_size": (256, 256),  # Detector size in pixels (height, width)
    "NA": 1.3
}

# Training Parameters
training_params = {
    "angles_number": 1,  # Number of azimuth angles per image
    "batch_size": 4,  # Batch size for training
    "epochs": 100,  # Number of epochs for training
    "learning_rate": 0.0001,  # Learning rate for optimizer #0.0001 worked for fringe images
    "validation_split": 0.2,  # 20% of data for validation
    "dataset size": 4000  #Total size of dataset (training+val+test)
}

# Define paths (Update these paths for your system)
dataset_path = None#[r"C:\Users\jw\Desktop\dyplomy\flowers_dataset"]

save_path = r"C:\Users\jw\Desktop\dyplomy\Erkosar Deniz\data\dataset.h5"  # Path to save the dataset

# TensorBoard Log Directory
log_dir = "logs/fit/" + tf.keras.callbacks.TensorBoard().log_dir
os.makedirs(log_dir, exist_ok=True)

# Number of additional epochs
additional_epochs = 50

model_path =\
    r"C:\Users\jw\Desktop\dyplomy\Erkosar Deniz\best_trained_models\two_outputs__amplitude_no_repetitions\model_checkpoint.keras"


try:
    model = tf.keras.models.load_model(model_path, custom_objects={'Lambda': Lambda(norm_vec)})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

if __name__ == "__main__":
    # Generate dataset if not already created
    if dataset_path and save_path:
        print("Generating dataset...")
        generate_training_data(exp_sys_params, training_params, dataset_path, save_path,
                               training_params["dataset size"])  # ,"fringes")

    # Load dataset in a memory-efficient way
    print("Loading dataset using preprocessed_dataset()...")
    train_dataset, validation_dataset, test_dataset = preprocessed_dataset(
        save_path, training_params["batch_size"], training_params["validation_split"]
    )

    # Create TensorBoard Callback
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_path = os.path.join(os.getcwd(), r'trained_model/model_checkpoint.keras')
    checkpoint = ModelCheckpoint(
        # os.path.join(os.getcwd(), r'trained_model/epoch_{epoch:02d}_model_checkpoint.keras'),
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True
    )
    # lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Train the model using the dataset
    train_data_size = (1.0 - training_params["validation_split"]) * training_params["dataset size"]
    steps_per_epoch = np.ceil(train_data_size / training_params["batch_size"]).astype(int)
    val_data_size = (training_params["validation_split"] / 2) * training_params["dataset size"]
    val_steps_per_epoch = np.ceil(val_data_size / training_params["batch_size"]).astype(int)
    test_steps_per_epoch = val_steps_per_epoch

    # Resume training
    history = model.fit(
        train_dataset,
        initial_epoch=training_params["epochs"],  # Start at epoch 100
        epochs=training_params["epochs"] + additional_epochs,  # Train until epoch 150
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=val_steps_per_epoch,
        callbacks=[tensorboard_callback, checkpoint],
        verbose=1
    )

