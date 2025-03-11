import numpy as np
import h5py
import os
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter, zoom
from numpy.fft import fftshift, ifft2, fft2


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


def generate_training_data(exp_sys_params, training_params, dataset_paths, save_path):
    """
    Generates synthetic training data for fringe pattern simulation, using multiple images and varying phase delays.

    Parameters:
        exp_sys_params (dict): Experimental system parameters.
        training_params (dict): Training-related parameters.
        dataset_paths (list): List of paths to folders containing images for phase objects.
        save_path (str): Path to save the generated dataset in HDF5 format.
    """

    # Initialize list to store image file paths
    image_files = []
    for dataset_path in dataset_paths:
        # Add images from each directory to the image_files list
        if os.path.exists(dataset_path):
            image_files.extend([
                os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
        else:
            print(f"Warning: {dataset_path} does not exist!")

    if not image_files:
        raise FileNotFoundError("No images found in the dataset directories.")

    # Define simulation parameters
    sampling_rate = exp_sys_params["pix_size"]
    x = np.arange(-exp_sys_params["detector_size"][0] / 2, exp_sys_params["detector_size"][0] / 2) * sampling_rate
    y = np.arange(-exp_sys_params["detector_size"][1] / 2, exp_sys_params["detector_size"][1] / 2) * sampling_rate
    x2d, y2d = np.meshgrid(x, y)

    angles_number = training_params["angles_number"]
    delta_ph_max = np.pi  # Maximum phase variation

    # Create HDF5 file for dataset storage
    with h5py.File(save_path, "w") as hf:
        dset_images = hf.create_dataset("input_images",
                                        (len(image_files) * angles_number, *exp_sys_params["detector_size"], 1),
                                        dtype="float32")
        dset_labels = hf.create_dataset("output_labels", (len(image_files) * angles_number, 1), dtype="float32")

        index = 0  # Track index in dataset
        for img_path in image_files:
            ph_obj = imageio.imread(img_path).astype(float)

            # Convert to grayscale if needed
            if ph_obj.ndim == 3:
                ph_obj = np.mean(ph_obj, axis=-1)

            # Normalize and resize phase map
            ph_obj = (ph_obj - np.min(ph_obj)) / (np.max(ph_obj) - np.min(ph_obj)) * delta_ph_max
            zoom_factor_y = exp_sys_params["detector_size"][1] / ph_obj.shape[0]
            zoom_factor_x = exp_sys_params["detector_size"][0] / ph_obj.shape[1]
            ph_obj = zoom(ph_obj, (zoom_factor_y, zoom_factor_x), order=1)

            # Generate the object wave
            u_obj = np.exp(1j * ph_obj)

            for _ in range(angles_number):
                beam_azimuth = np.random.uniform(0, 2 * np.pi)  # Random azimuth angle

                # Compute spatial frequencies for object tilt
                fc_xy = [
                    np.cos(beam_azimuth) * np.sin(exp_sys_params["beam_tilt_angle"]) * (
                                exp_sys_params["ri_immersion"] / exp_sys_params["wavelength"]),
                    np.sin(beam_azimuth) * np.sin(exp_sys_params["beam_tilt_angle"]) * (
                                exp_sys_params["ri_immersion"] / exp_sys_params["wavelength"]),
                ]

                # Object wave and Fourier mask for limited NA
                NA = 1.3  # Numerical aperture
                fillx = exp_sys_params["ri_immersion"] * np.sin(exp_sys_params["beam_tilt_angle"]) * np.cos(
                    beam_azimuth) / exp_sys_params["wavelength"]
                filly = exp_sys_params["ri_immersion"] * np.sin(exp_sys_params["beam_tilt_angle"]) * np.sin(
                    beam_azimuth) / exp_sys_params["wavelength"]
                fNA = NA / exp_sys_params["wavelength"]
                dfx = 1 / (exp_sys_params["detector_size"][0] * sampling_rate)
                dfy = 1 / (exp_sys_params["detector_size"][1] * sampling_rate)
                fx = np.arange(-exp_sys_params["detector_size"][0] / 2, exp_sys_params["detector_size"][0] / 2) * dfx
                fy = np.arange(-exp_sys_params["detector_size"][1] / 2, exp_sys_params["detector_size"][1] / 2) * dfy
                Fx, Fy = np.meshgrid(fx, fy)

                # Create Fourier mask
                ft_mask = ((Fx - fillx) ** 2 + (Fy - filly) ** 2 < fNA ** 2).astype(float)
                ft_mask = gaussian_filter(ft_mask, sigma=10, mode='constant')  # Smooth mask to avoid Gibbs oscillations

                # Apply Fourier mask
                ftu_obj_na = fftshift(fft2(fftshift(u_obj))) * ft_mask
                u_obj_na = fftshift(ifft2(fftshift(ftu_obj_na)))

                object_amp = np.abs(u_obj_na)

                # # Compute object beam with object influence
                # object_beam = np.exp(1j * 2 * np.pi * (x2d * fc_xy[0] + y2d * fc_xy[1]))
                # object_beam = u_obj_na * object_beam

                # Generate fringe pattern
                #fringe_image = np.power(np.abs(object_beam + 1.0), 2)  # Reference beam = 1.0
                dset_images[index, :, :, 0] = normalize(object_amp)
                dset_labels[index, 0] = beam_azimuth

                index += 1

    print(f"Dataset saved to {save_path}")
