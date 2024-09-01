from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

matplotlib.use("tkagg")


def gauss_fun(x, x0, sigma):
    return np.exp(-np.power(x - x0, 2) / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)


def generate_gaussians_set(sigma):
    gaussian_set = np.zeros((256, 256))
    for i in range(256):
        gaussian_set[i, :] = gauss_fun(np.arange(256), i, sigma)
    return gaussian_set


def generate_gaussian_3d(ix, iy, iz, gaussian_set):
    return np.tensordot(
        np.tensordot(gaussian_set[ix, :], gaussian_set[iy, :], axes=0),
        gaussian_set[iz, :],
        axes=0,
    )


if __name__ == "__main__":
    rgb = np.asarray(Image.open("picture.jpg"))

    flag_max_count_achieved = False

    rgb_list = np.reshape(np.reshape(rgb, (1, -1, 3)), (-1, 3))

    rgb_hist = np.zeros((256, 256, 256))

    for pix in tqdm(rgb_list):
        rgb_hist[pix[0], pix[1], pix[2]] += 1

    blurred = np.copy(rgb_hist)
    while not flag_max_count_achieved:
        # gaussian_set=generate_gaussians_set(blur_size)

        max_count = np.sum(
            np.logical_and(
                np.logical_and(
                    np.logical_and(
                        np.logical_and(
                            np.logical_and(
                                blurred[1:-1, 1:-1, 1:-1] > blurred[0:-2, 1:-1, 1:-1],
                                blurred[1:-1, 1:-1, 1:-1] > blurred[2:, 1:-1, 1:-1],
                            ),
                            blurred[1:-1, 1:-1, 1:-1] > blurred[1:-1, 0:-2, 1:-1],
                        ),
                        blurred[1:-1, 1:-1, 1:-1] > blurred[1:-1, 2:, 1:-1],
                    ),
                    blurred[1:-1, 1:-1, 1:-1] > blurred[1:-1, 1:-1, 0:-2],
                ),
                blurred[1:-1, 1:-1, 1:-1] > blurred[1:-1, 1:-1, 2:],
            )
        )

        if max_count <= 25:
            break

        blurred = np.divide(blurred, np.sum(blurred))

        pad_blurred = np.pad(blurred, ((1, 1), (1, 1), (1, 1)))
        laplacian = (
            6 * pad_blurred[1:-1, 1:-1, 1:-1]
            - pad_blurred[0:-2, 1:-1, 1:-1]
            - pad_blurred[2:, 1:-1, 1:-1]
            - pad_blurred[1:-1, 0:-2, 1:-1]
            - pad_blurred[1:-1, 2:, 1:-1]
            - pad_blurred[1:-1, 1:-1, 0:-2]
            - pad_blurred[1:-1, 1:-1, 2:]
        )

        blurred -= laplacian * 0.1

        print(max_count)

    print("stop here")
