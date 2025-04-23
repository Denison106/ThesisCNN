import os

import matplotlib.pyplot as plt
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from scipy.io import loadmat


def load_dataset(save_path):

    try:
        h5f = h5py.File(save_path, "r")

        if 'inputs' not in h5f or 'targets' not in h5f:
            raise KeyError("Missing 'inputs' or 'targets' datasets in the HDF5 file.")

        dataset = {"inputs": h5f['inputs'], "targets": h5f['targets']} # use lazy loading to create an object

        num_samples = dataset['inputs'].shape[0]
        print(f"Loaded dataset: {num_samples} samples into dataset.")
        return dataset

    except FileNotFoundError:
        raise FileNotFoundError("The dataset file was not found.")
    except KeyError as e:
        raise KeyError(f"Missing expected data key in the file: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the dataset: {e}")


def data_generator(inputs, targets, batch_size):
    total_samples = inputs.shape[0]

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
    dataset = load_dataset(save_path)

    inputs = dataset["inputs"]
    targets = dataset["targets"]

    # Calculate set size
    total_samples = len(inputs)
    val_test_size = int(total_samples * validation_split)
    val_size = val_test_size // 2
    train_size = total_samples - val_test_size

    # Calculate the indices for dataset splitting
    train_indices = range(0, train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, total_samples)

    # Define generators
    train_gen = lambda: data_generator(inputs[train_indices], targets[train_indices], batch_size)
    val_gen = lambda: data_generator(inputs[val_indices], targets[val_indices], batch_size)
    test_gen = lambda: data_generator(inputs[test_indices], targets[test_indices], batch_size)

    train_dataset = Dataset.from_generator(
        train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None,) + inputs.shape[1:], dtype=tf.float32),
            tf.TensorSpec(shape=(None,) + targets.shape[1:], dtype=tf.float32),
        )
    )

    validation_dataset = Dataset.from_generator(
        val_gen,
        output_signature=(
            tf.TensorSpec(shape=(None,) + inputs.shape[1:], dtype=tf.float32),
            tf.TensorSpec(shape=(None,) + targets.shape[1:], dtype=tf.float32),
        )
    )

    test_dataset = Dataset.from_generator(
        test_gen,
        output_signature=(
            tf.TensorSpec(shape=(None,) + inputs.shape[1:], dtype=tf.float32),
            tf.TensorSpec(shape=(None,) + targets.shape[1:], dtype=tf.float32),
        )
    )
    train_dataset = train_dataset.shuffle(1024).repeat().prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.repeat().prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset


def load_dataset_to_tensors(save_path, batch_size, validation_split):
    """
    Loads dataset in a memory-efficient way using TensorFlow `Dataset` API.

    Parameters:
        save_path (str): Path to the HDF5 dataset.
        batch_size (int): The batch size for training.
        validation_split (float): Percentage of data to use for validation.

    Returns:
        train_dataset, validation_dataset, test_dataset: TensorFlow dataset objects.
    """
    dataset = load_dataset(save_path)

    inputs = dataset["inputs"]
    targets = dataset["targets"]

    # Calculate set size
    total_samples = len(inputs)
    val_test_size = int(total_samples * validation_split)
    val_size = val_test_size // 2
    train_size = total_samples - val_test_size

    # Calculate the indices for dataset splitting
    train_indices = range(0, train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, total_samples)

    # Define generators
    train_data = inputs[train_indices]
    train_labels = targets[train_indices]
    val_data = inputs[test_indices]
    val_labels = targets[test_indices]
    test_data = inputs[test_indices]
    test_labels = targets[test_indices]

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def visualize_data(input_images, target_labels, display_params):
    """
    Visualizes the simulated fringe pattern data.
    Parameters:
        input_images (ndarray): Array of simulated fringe pattern images.
        target_labels (ndarray): Array of simulated azimuths.
        display_params (dict): Display-related parameters.
    """
    plt.ion()
    for i in range(input_images.shape[0]):
        plt.imshow(input_images[i, :, :, 0], cmap="viridis")
        title_txt = "azimuth = {:.2f} deg"
        azimuth_deg = np.rad2deg(target_labels[i, 0])
        plt.title(title_txt.format(azimuth_deg))
        plt.colorbar()
        plt.show()
        plt.pause(display_params["display_pause_time"])
        plt.clf()
    plt.ioff()
    plt.close()


def plot_learning_curves(history):
    """
    Plots the learning curves (loss and MAE) from the training history.

    Parameters:
        history: Training history object returned by the Keras model's `fit` method.
    """
    plt.figure()

    # try:
    #     # Try to access and plot loss per iteration
    #     with open("trained_model/train_losses", "rb") as fp:
    #         train_loss_per_batch = pickle.load(fp)
    #     iters = range(1, len(train_loss_per_batch) + 1)
    #     plt.plot(iters, train_loss_per_batch, 'b', label='Training Loss')
    # except:
    #     print("An exception occurred")

 #  steps_per_epoch = (len(train_loss_per_batch)+1)/(len(history.history['loss']) + 1)
    epochs = np.array(range(len(history.history['loss'])))+1
    plt.plot(epochs, history.history['loss'], 'bo', label='Loss')
    plt.plot(epochs, history.history['val_loss'], 'ro', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(history.history['mae'], label='Mean Absolute Error', color='red')
    plt.title('Mean Absolute Error Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()


def normalize(img):
    """
    Normalizes an image to the range [0, 1].

    Parameters:
        img (ndarray): Input image.

    Returns:
        ndarray: Normalized image with values in the range [0, 1].
    """
    min_val = np.min(img)
    max_val = np.max(img)
    return (img - min_val) / (max_val - min_val)

def load_sino_and_azim(mat_path):
    sino = None
    azim_vec = None

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"File not found: {mat_path}")

    try:
        # Try loading using h5py (works for MATLAB v7.3+)
        with h5py.File(mat_path, 'r') as f:
            # Load sino
            if 'sino' in f:
                sino = np.array(f['sino']).astype("complex")
            else:
                raise KeyError("Missing 'sino' in HDF5 .mat file")

            # Load azim_vec if it exists
            if 'azim_vec' in f:
                azim_vec = np.array(f['azim_vec'])
                azim_vec = np.squeeze(azim_vec)

    except (OSError, IOError):
        # Fallback to scipy for older MATLAB files
        try:
            mat = loadmat(mat_path)
            if 'sino' in mat:
                sino = mat['sino'].astype("complex")
            else:
                raise KeyError("Missing 'sino' in legacy .mat file")

            if 'azim_vec' in mat:
                azim_vec = mat['azim_vec']
                azim_vec = np.squeeze(azim_vec)

        except Exception as e:
            raise RuntimeError(f"Failed to load .mat file: {e}")

    return sino, azim_vec


def test_model(model, sino, display, gt_azimuth=None):

    img_no = sino.shape[-1]
    print(img_no)
    if gt_azimuth:
        gt_azimuth = gt_azimuth % np.pi

    pred_azimuth = np.empty((img_no,), dtype="float32")

    # plt.ion()
    sino = np.abs(sino)  # make sure that work on amplitude images
    for i in range(img_no):
        current_im = sino[..., i]
        current_im = normalize(current_im)  # preprocess data as it was done for taining data!
        pred_azimuth[i] = model.predict(tf.expand_dims(current_im, axis=0)).squeeze()
        if display:
            plt.figure()
            plt.imshow(current_im, cmap="viridis")
            if gt_azimuth:
                title_txt = "gt: a={:.2f} \n pred: a={:.2f}"
                formatted_title_txt = title_txt.format(np.rad2deg(gt_azimuth[i]),
                                                       np.rad2deg(pred_azimuth[i]))
            else:
                title_txt = "pred: a={:.2f}"
                formatted_title_txt = title_txt.format(np.rad2deg(pred_azimuth[i]))

            plt.title(formatted_title_txt)
            plt.colorbar()
        #     plt.pause(2)
        #     plt.clf()
        # plt.ioff()
        # plt.close()
    plt.show()

    outputs = [pred_azimuth]
    if gt_azimuth is not None:
        outputs.extend([gt_azimuth])
    return tuple(outputs)
