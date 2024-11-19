import numpy as np
import matplotlib.pyplot as plt

def normalize(img):
    min_val = np.min(img)
    max_val = np.max(img)
    return (img - min_val) / (max_val - min_val)


if __name__ == '__main__':

    # All distances are in um
    pix_size = 3.5
    optical_mag = 1000
    beam_tilt_angle = np.deg2rad(40)
    angles_number = 10
    beam_azimuth_vec = 2 * np.pi*np.random.rand(angles_number,1)
    detector_size = (512, 512)
    wavelength = 0.5
    ri_immersion = 1.33
    ri_bead = 1.34
    inputTrainImages = np.empty((detector_size[0], detector_size[1], 1, angles_number), dtype=float)
    outputTrainLabels = np.empty((angles_number, angles_number), dtype=float)
    sampling_rate = pix_size / optical_mag
    x = np.arange(-detector_size[0] / 2, detector_size[0] / 2) * sampling_rate
    y = np.arange(-detector_size[1] / 2, detector_size[1] / 2) * sampling_rate
    [x2d, y2d] = np.meshgrid(x, y)
    reference_beam = 1.0

    plt.ion()

    for data_no, beam_azimuth in enumerate(beam_azimuth_vec):
        fc_xy = [np.cos(beam_azimuth) * np.sin(beam_tilt_angle) * (ri_immersion / wavelength),
                 np.sin(beam_azimuth) * np.sin(beam_tilt_angle) * (ri_immersion / wavelength)]

        object_beam = np.exp(1j * 2 * np.pi * (x2d * fc_xy[0] + y2d * fc_xy[1]))
        fringe_image = np.power(np.abs(object_beam + reference_beam), 2)
        fringe_image = normalize(fringe_image)

        inputTrainImages[:,:,0,data_no] = fringe_image
        outputTrainLabels[:,data_no] = beam_azimuth

        # Change image contents
        plt.imshow(fringe_image)
        plt.show()
        plt.colorbar()
        plt.pause(1)
        plt.clf()