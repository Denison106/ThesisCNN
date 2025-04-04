# TODO: The network learns from fringe images but not from the corresponding amplitude images.
# TODO: Using fewer flower images and more azimuth angles per image helps—further analysis is needed.
# TODO: JW – change the output to sin(azimuth) and cos(azimuth), and replace the sigmoid output with a linear output.
# TODO: Maybe choose easier parameters, e.g., steeper illumination.


import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
import pandas as pd

from data_generator import generate_training_data
from network import build_winnik_model, build_deniz_model
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
    "epochs": 20*8,  # Number of epochs for training
    "learning_rate": 0.0001,  # Learning rate for optimizer #0.0001 worked for fringe images
    "validation_split": 0.2,  # 20% of data for validation
    "dataset size": 4000  #Total size of dataset (training+val+test)
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
        generate_training_data(exp_sys_params, training_params, dataset_path, save_path, training_params["dataset size"])#,"fringes")

    # Load dataset in a memory-efficient way
    print("Loading dataset using preprocessed_dataset()...")
    train_dataset, validation_dataset, test_dataset = preprocessed_dataset(
        save_path, training_params["batch_size"], training_params["validation_split"]
    )

    # train_data, train_labels, val_data, val_labels, test_data, test_labels = load_dataset_to_tensors(
    #     save_path, training_params["batch_size"], training_params["validation_split"])
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(
    #     buffer_size=train_data.shape[0]).batch(training_params["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)
    # validation_dataset=(val_data, val_labels)
    # test_dataset = test_data# test_labels)

    # Build the CNN model
    model = build_winnik_model(input_shape=exp_sys_params["detector_size"]+(1,),
                               learning_rate=training_params["learning_rate"])

    # Create TensorBoard Callback
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_path = os.path.join(os.getcwd(), r'trained_model/model_checkpoint.keras')
    checkpoint = ModelCheckpoint(
        #os.path.join(os.getcwd(), r'trained_model/epoch_{epoch:02d}_model_checkpoint.keras'),
        checkpoint_path,
        monitor = "val_loss",
        save_best_only=True
    )
    #lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

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
        callbacks=[tensorboard_callback, checkpoint],  # Include TensorBoard Callback
        verbose=1
    )

    # # Model evaluation
    # im = val_data[0:10]
    # im = tf.expand_dims(im, axis=0) if im.shape[0] == val_data.shape[1] else im
    # predicted = model.predict(im) #* np.pi
    # errors = np.abs(val_labels[0:10, 0] - predicted[:, 0])
    #
    # # Display results
    # df = pd.DataFrame({
    #     'Original': val_labels[0:10, 0],
    #     'Predicted': predicted[:, 0],
    #     'AbsError': errors,
    #     'Error %': (errors * 100 / np.pi)
    # })
    # print(df)

    score = model.evaluate(test_dataset,
                           batch_size=training_params["batch_size"],
                           steps=test_steps_per_epoch,
                           verbose=1)

    print(f'Test loss: {score}')

    # Plot learning curves
    plot_learning_curves(history)

    print(f"Training completed. Run TensorBoard with: tensorboard --logdir={log_dir}")

    num_of_batches = 2
    test_data_iterator = iter(test_dataset.batch(num_of_batches))
    tensor_batch = next(test_data_iterator)  # Get batches
    im = tf.reshape(tensor_batch[0],
                    (num_of_batches * training_params["batch_size"], *exp_sys_params["detector_size"], 1))
    labels = tf.reshape(tensor_batch[1], (num_of_batches * training_params["batch_size"], 2))
    predicted_labels = model.predict(im)

    # Display results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    df = pd.DataFrame({
        'Original cos(a)': labels[:, 0],
        'Predicted cos(a)': predicted_labels[:, 0],
        'Error cos(a)': np.abs(predicted_labels[:, 0] - labels[:, 0]),
        'Original sin(a)': labels[:, 1],
        'Predicted sin(a)': predicted_labels[:, 1],
        'Error sin(a)': np.abs(predicted_labels[:, 1] - labels[:, 1]),
    })
    print(df)
    print("\n Mean values:")
    print(df.mean())

    for i in range(im.shape[0]):
        current_im = im[i, ...]
        predicted_lab = model.predict(tf.expand_dims(current_im, axis=0))
        plt.figure()
        plt.imshow(current_im, cmap="viridis")
        title_txt = "gt: ca = {:.2f}; sa = {:.2f}; a={:.2f} \n pred: ca = {:.2f}; sa = {:.2f}; a={:.2f}"
        gt_azimuth = np.rad2deg(np.arctan2(labels[i, 1], labels[i, 0]))
        pred_azimuth = np.rad2deg(np.arctan2(predicted_lab[0, 1],  predicted_lab[0, 0]))
        plt.title(title_txt.format(labels[i, 0], labels[i, 1], gt_azimuth,
                                   predicted_lab[0, 0], predicted_lab[0, 1], pred_azimuth))
        plt.colorbar()

    plt.show()


