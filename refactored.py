import numpy as np
import matplotlib.pyplot as plt

# Configuration dictionary
config = {
    "pix_size": 3.5,
    "optical_mag": 1000,
    "beam_tilt_angle": np.deg2rad(40),
    "angles_number": 10,
    "detector_size": (512, 512),
    "wavelength": 0.5,
    "ri_immersion": 1.33,
    "ri_bead": 1.34,
    "display_pause_time": 1,  # Time in seconds to display each image
}

# Function to normalize an image
def normalize(img):
    min_val = np.min(img)
    max_val = np.max(img)
    return (img - min_val) / (max_val - min_val)

# Function to generate training data
def generate_training_data(config):
    sampling_rate = config["pix_size"] / config["optical_mag"]
    x = np.arange(-config["detector_size"][0] / 2, config["detector_size"][0] / 2) * sampling_rate
    y = np.arange(-config["detector_size"][1] / 2, config["detector_size"][1] / 2) * sampling_rate
    x2d, y2d = np.meshgrid(x, y)

    input_images = np.empty((config["detector_size"][0], config["detector_size"][1], 1, config["angles_number"]), dtype=float)
    output_labels = np.empty(config["angles_number"], dtype=float)
    beam_azimuth_vec = 2 * np.pi * np.random.rand(config["angles_number"], 1)
    reference_beam = 1.0

    for i, beam_azimuth in enumerate(beam_azimuth_vec):
        fc_xy = [
            np.cos(beam_azimuth) * np.sin(config["beam_tilt_angle"]) * (config["ri_immersion"] / config["wavelength"]),
            np.sin(beam_azimuth) * np.sin(config["beam_tilt_angle"]) * (config["ri_immersion"] / config["wavelength"])
        ]
        object_beam = np.exp(1j * 2 * np.pi * (x2d * fc_xy[0] + y2d * fc_xy[1]))
        fringe_image = np.power(np.abs(object_beam + reference_beam), 2)
        input_images[:, :, 0, i] = normalize(fringe_image)
        output_labels[i] = beam_azimuth[0]  # Extract scalar value

    return input_images, output_labels


# Function to visualize data
def visualize_data(input_images, config):
    plt.ion()
    for i in range(input_images.shape[-1]):
        plt.imshow(input_images[:, :, 0, i], cmap='viridis')
        plt.colorbar()
        plt.show()
        plt.pause(config["display_pause_time"])
        plt.clf()

# Main execution block
if __name__ == "__main__":
    input_images, output_labels = generate_training_data(config)
    visualize_data(input_images, config)
