import cv2


def createHOGDescriptor(win_size, block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8), n_bins=9,
                        signed_gradients=False):
    deriv_aperture = 1
    win_sigma = -1
    histogram_norm_type = 0
    l2_hys_threshold = 0.2
    gamma_correction = 1
    n_levels = 64

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins, deriv_aperture, win_sigma,
                            histogram_norm_type, l2_hys_threshold, gamma_correction, n_levels, signed_gradients)

    return hog
