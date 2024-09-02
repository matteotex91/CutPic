from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

matplotlib.use("tkagg")


def count_maxima(array: np.ndarray):
    return np.sum(
        np.logical_and(
            np.logical_and(
                np.logical_and(
                    np.logical_and(
                        np.logical_and(
                            array[1:-1, 1:-1, 1:-1] > array[0:-2, 1:-1, 1:-1],
                            array[1:-1, 1:-1, 1:-1] > array[2:, 1:-1, 1:-1],
                        ),
                        array[1:-1, 1:-1, 1:-1] > array[1:-1, 0:-2, 1:-1],
                    ),
                    array[1:-1, 1:-1, 1:-1] > array[1:-1, 2:, 1:-1],
                ),
                array[1:-1, 1:-1, 1:-1] > array[1:-1, 1:-1, 0:-2],
            ),
            array[1:-1, 1:-1, 1:-1] > array[1:-1, 1:-1, 2:],
        )
    )


def padded_laplacian(array: np.ndarray):
    padded_array = np.pad(array, ((1, 1), (1, 1), (1, 1)), "edge")
    return (
        6 * padded_array[1:-1, 1:-1, 1:-1]
        - padded_array[0:-2, 1:-1, 1:-1]
        - padded_array[2:, 1:-1, 1:-1]
        - padded_array[1:-1, 0:-2, 1:-1]
        - padded_array[1:-1, 2:, 1:-1]
        - padded_array[1:-1, 1:-1, 0:-2]
        - padded_array[1:-1, 1:-1, 2:]
    )


if __name__ == "__main__":
    rgb = np.asarray(Image.open("picture.jpg"))

    flag_max_count_achieved = False

    rgb_list = np.reshape(np.reshape(rgb, (1, -1, 3)), (-1, 3))

    rgb_hist = np.zeros((256, 256, 256))

    np.random.rand

    for pix in tqdm(rgb_list):
        rgb_hist[pix[0], pix[1], pix[2]] += 1

    blurred_rgb_hist = np.copy(rgb_hist)
    while not flag_max_count_achieved:
        max_count = count_maxima(blurred_rgb_hist)

        if max_count <= 25:
            break

        blurred_rgb_hist = np.divide(blurred_rgb_hist, np.sum(blurred_rgb_hist))

        blurred_rgb_hist -= padded_laplacian(blurred_rgb_hist) * 0.1

        print(max_count)

    print("stop here")
