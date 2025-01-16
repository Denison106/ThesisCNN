import numpy as np
from scipy.io import loadmat
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


def generate_training_data(exp_sys_params, training_params):
    """
    Generates synthetic training data for fringe pattern simulation, including object influence.

    Parameters:
        exp_sys_params (dict): Experimental system parameters.
        training_params (dict): Training-related parameters.

    Returns:
        tuple: A tuple containing:
            - input_images (ndarray): Simulated fringe pattern images in the format (batch, height, width, channels).
            - output_labels (ndarray): Corresponding azimuth angles.
    """
    # Set delta_ph to pi or 2pi 
    delta_ph = np.pi  # Maximum phase variation
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

    # Load object phase map
    mat = loadmat('my_coin.mat')  # Ensure 'my_coin.mat' is in the working directory
    ph_obj = mat['img']

    # Normalize and resize phase map
    ph_obj = (ph_obj - ph_obj.min()) / (ph_obj.max() - ph_obj.min()) * delta_ph
    zoom_factor_y = exp_sys_params["detector_size"][1] / ph_obj.shape[0]
    zoom_factor_x = exp_sys_params["detector_size"][0] / ph_obj.shape[1]
    ph_obj = zoom(ph_obj, (zoom_factor_y, zoom_factor_x), order=1)

    # Generate the object wave
    u_obj = np.exp(1j * ph_obj)

    for i, beam_azimuth in enumerate(beam_azimuth_vec):
        # Compute spatial frequencies for object tilt
        fc_xy = [
            np.cos(beam_azimuth) * np.sin(exp_sys_params["beam_tilt_angle"]) * (exp_sys_params["ri_immersion"] / exp_sys_params["wavelength"]),
            np.sin(beam_azimuth) * np.sin(exp_sys_params["beam_tilt_angle"]) * (exp_sys_params["ri_immersion"] / exp_sys_params["wavelength"]),
        ]

        # Object wave and Fourier mask for limited NA
        NA = 1.3  # Numerical aperture
        fillx = exp_sys_params["ri_immersion"] * np.sin(exp_sys_params["beam_tilt_angle"]) * np.cos(beam_azimuth) / exp_sys_params["wavelength"]
        filly = exp_sys_params["ri_immersion"] * np.sin(exp_sys_params["beam_tilt_angle"]) * np.sin(beam_azimuth) / exp_sys_params["wavelength"]
        fNA = NA / exp_sys_params["wavelength"]
        dfx = 1 / (exp_sys_params["detector_size"][0] * sampling_rate)
        dfy = 1 / (exp_sys_params["detector_size"][1] * sampling_rate)
        fx = np.arange(-exp_sys_params["detector_size"][0] / 2, exp_sys_params["detector_size"][0] / 2) * dfx
        fy = np.arange(-exp_sys_params["detector_size"][1] / 2, exp_sys_params["detector_size"][1] / 2) * dfy
        Fx, Fy = np.meshgrid(fx, fy)

        # Create Fourier mask (no padding applied here)
        ft_mask = ((Fx - fillx)**2 + (Fy - filly)**2 < fNA**2).astype(float)
        ft_mask = gaussian_filter(ft_mask, sigma=10, mode='constant')  # Smooth the mask to avoid Gibbs oscillations

        # Apply the mask in Fourier space
        ftu_obj_na = fftshift(fft2(fftshift(u_obj))) * ft_mask
        u_obj_na = fftshift(ifft2(fftshift(ftu_obj_na)))

        # Compute the object beam with object influence
        object_beam = np.exp(1j * 2 * np.pi * (x2d * fc_xy[0] + y2d * fc_xy[1]))
        object_beam = u_obj_na * object_beam

        # Generate the fringe pattern
        fringe_image = np.power(np.abs(object_beam + reference_beam), 2)
        input_images[i, :, :, 0] = normalize(fringe_image)
        output_labels[i] = beam_azimuth[0]  # Extract scalar value

    return input_images, output_labels
