import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Experimental System Parameters (distances in micrometers [um])
exp_sys_params = {
    "pix_size": 3.5,  # Pixel size in um
    "optical_mag": 100,  # Optical magnification (realistic value)
    "wavelength": 0.5,  # Wavelength of light in um
    "ri_immersion": 1.33,  # Refractive index of immersion medium
    "beam_tilt_angle": np.deg2rad(40),  # Beam tilt angle in radians
    "detector_size": (512, 512),  # Detector size in pixels
}

# Simulated Object Parameters (currently unused in the simulation)
sim_obj_params = {
    "object_present": False,  # Whether a simulated object is included
}

# Display Parameters
display_params = {
    "display_pause_time": 1,  # Pause time for visualization in seconds
}

# Training Parameters
training_params = {
    "angles_number": 10,  # Number of azimuth angles to simulate
    "batch_size": 4,  # Batch size for training
    "epochs": 5,  # Number of epochs for training
    "learning_rate": 0.001,  # Learning rate for optimizer
}


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


def generate_training_data(exp_sys_params, training_params):
    """
    Generates synthetic training data for fringe pattern simulation.

    Parameters:
        exp_sys_params (dict): Experimental system parameters.
        training_params (dict): Training-related parameters.

    Returns:
        tuple: A tuple containing:
            - input_images (ndarray): Simulated fringe pattern images in the format (batch, height, width, channels).
            - output_labels (ndarray): Corresponding azimuth angles.
    """
    sampling_rate = exp_sys_params["pix_size"] / exp_sys_params["optical_mag"]
    x = np.arange(-exp_sys_params["detector_size"][0] / 2, exp_sys_params["detector_size"][0] / 2) * sampling_rate
    y = np.arange(-exp_sys_params["detector_size"][1] / 2, exp_sys_params["detector_size"][1] / 2) * sampling_rate
    x2d, y2d = np.meshgrid(x, y)

    input_images = np.empty(
        (training_params["angles_number"], exp_sys_params["detector_size"][0], exp_sys_params["detector_size"][1], 1),
        dtype=float,
    )
    output_labels = np.empty((training_params["angles_number"], 1), dtype=float)
    beam_azimuth_vec = 2 * np.pi * np.random.rand(training_params["angles_number"], 1)
    reference_beam = 1.0

    for i, beam_azimuth in enumerate(beam_azimuth_vec):
        fc_xy = [
            np.cos(beam_azimuth) * np.sin(exp_sys_params["beam_tilt_angle"]) * (exp_sys_params["ri_immersion"] / exp_sys_params["wavelength"]),
            np.sin(beam_azimuth) * np.sin(exp_sys_params["beam_tilt_angle"]) * (exp_sys_params["ri_immersion"] / exp_sys_params["wavelength"]),
        ]
        object_beam = np.exp(1j * 2 * np.pi * (x2d * fc_xy[0] + y2d * fc_xy[1]))
        fringe_image = np.power(np.abs(object_beam + reference_beam), 2)
        input_images[i, :, :, 0] = normalize(fringe_image)
        output_labels[i] = beam_azimuth[0]  # Extract scalar value

    return input_images, output_labels


def visualize_data(input_images, display_params):
    """
    Visualizes the simulated fringe pattern data.

    Parameters:
        input_images (ndarray): Array of simulated fringe pattern images.
        display_params (dict): Display-related parameters.
    """
    plt.ion()
    for i in range(input_images.shape[0]):
        plt.imshow(input_images[i, :, :, 0], cmap="viridis")
        plt.colorbar()
        plt.show()
        plt.pause(display_params["display_pause_time"])
        plt.clf()


def build_model(input_shape):
    """
    Builds a simple CNN model for azimuth angle prediction.

    Parameters:
        input_shape (tuple): Shape of the input data (height, width, channels).

    Returns:
        Model: Compiled Keras model.
    """
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')  # Predict azimuth angle directly
    ])
    model.compile(optimizer=Adam(learning_rate=training_params["learning_rate"]),
                  loss=MeanSquaredError(),
                  metrics=['mae'])
    return model


if __name__ == "__main__":
    # Generate synthetic data
    input_images, output_labels = generate_training_data(exp_sys_params, training_params)

    # Visualize the generated data
    visualize_data(input_images, display_params)

    # Build and train the model
    model = build_model(input_shape=input_images.shape[1:])
    history = model.fit(input_images, output_labels, batch_size=training_params["batch_size"], epochs=training_params["epochs"])

    # Plot the learning curve
    plt.figure()
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['mae'], label='Mean Absolute Error')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
