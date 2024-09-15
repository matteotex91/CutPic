from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from numpy.linalg import norm
import matplotlib as mpl
import pickle

import os


"""
problem with this method : no informations on the spatial correlation of the pixels
"""

COLOR_RESOLUTION = 256
COLOR_DISCRETIZATION = 24
RELAXATION_SPEED = 0.1

"""Produces a map of boolean having the same shape as the argument array, which is three dimensional.
Each element of this list is true if all the neighbors are smaller, false otherwise.
"""


def get_peaks_bool_map(array: np.ndarray):
    return np.logical_and(
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


def gradients(rgb: np.ndarray) -> np.ndarray:
    padded = np.pad(rgb, ((2, 2), (2, 2), (0, 0)), "edge")
    return np.linalg.norm(
        (
            np.abs(padded[2:, 1:-1, :] - padded[1:-1, 1:-1, :])
            + np.abs(padded[:-2, 1:-1, :] - padded[1:-1, 1:-1, :])
            + np.abs(padded[1:-1, 2:, :] - padded[1:-1, 1:-1, :])
            + np.abs(padded[1:-1, :-2, :] - padded[1:-1, 1:-1, :])
        ),
        axis=2,
    )


WIDTH = 1


def hist_weight_from_gradient(gradient):
    return np.exp(-((gradient / WIDTH) ** 2))


""" This function counts the peaks of the argument array, based on the get_peaks_bool_map function
"""


def get_peaks_count(array: np.ndarray):
    return np.sum(get_peaks_bool_map(array))


""" This function gives a matrix having shape (N,3), where N is just the peaks count. Each row of this matrix is the color assigned to the peak
"""


def get_peaks_color(array: np.ndarray):
    return np.stack(np.where(get_peaks_bool_map(array))).T


""" This function calculates the laplacian of the 3D array in the RGB space. The borders are considered by padding the array with the elements at the edge.
"""


def edge_padded_laplacian(array: np.ndarray):
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


def plot_peaks_reduction_timetrace(times_peaks):
    plt.scatter(times_peaks[:, 0], times_peaks[:, 1])
    plt.xlabel("time [s]")
    plt.ylabel("peaks count [-]")
    plt.grid(True)
    plt.yscale("log")
    plt.show()


def plot_sliced_image(color_map, sliced_image):
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "custom cmap", color_map, np.shape(color_map)[0]
    )
    plt.pcolormesh(sliced_image, cmap=cmap)
    plt.show()


def assign_closer_color(original_image, color_list):
    return np.array(
        [
            [
                np.argmin(
                    [
                        norm(np.array(original_image[i, j, :]) - col)
                        for col in color_list
                    ]
                )
                for i in range(original_image.shape[0])
            ]
            for j in tqdm(range(original_image.shape[1]))
        ]
    )


if __name__ == "__main__":
    pic_path = os.getcwd() + "/picture.jpg"
    folder = os.getcwd() + "/gradient_weight_voronoi/new/"
    preprocess_path = os.getcwd() + "/gradient_weight_voronoi/new/preprocess.pickle"
    rgb = np.asarray(Image.open(pic_path))

    if os.path.exists(preprocess_path):
        with open(preprocess_path, "rb") as f:
            sliced_image, color_map = pickle.load(f)
    else:

        flag_max_count_achieved = False

        grad_list = np.reshape(gradients(rgb), (-1, 1))

        rgb_list = np.reshape(np.reshape(rgb, (1, -1, 3)), (-1, 3))

        rgb_hist = np.zeros((COLOR_RESOLUTION, COLOR_RESOLUTION, COLOR_RESOLUTION))

        for pix, grad in tqdm(zip(rgb_list, grad_list)):
            rgb_hist[tuple(pix)] += hist_weight_from_gradient(grad)

        blurred_rgb_hist = np.copy(rgb_hist)
        backup_hist = np.copy(rgb_hist)
        times_peaks_list = []
        t0 = time()

        while not flag_max_count_achieved:
            peaks_count = get_peaks_count(blurred_rgb_hist)

            if peaks_count <= COLOR_DISCRETIZATION:
                break

            blurred_rgb_hist = np.divide(blurred_rgb_hist, np.sum(blurred_rgb_hist))

            backup_hist = np.copy(blurred_rgb_hist)

            blurred_rgb_hist -= (
                edge_padded_laplacian(blurred_rgb_hist) * RELAXATION_SPEED
            )

            print(peaks_count)
            times_peaks_list.append(np.array([time() - t0, peaks_count]))

        plot_peaks_reduction_timetrace(np.array(times_peaks_list))

        final_hist = (
            backup_hist if peaks_count < COLOR_DISCRETIZATION else blurred_rgb_hist
        )

        color_map = get_peaks_color(final_hist)

        hist_values = np.array([final_hist[tuple(c)] for c in color_map])

        sorted_color_map = np.array(
            [c for _, c in sorted(zip(hist_values, color_map))]
        )[-COLOR_DISCRETIZATION:]

        sliced_image = assign_closer_color(rgb, sorted_color_map)

        with open(preprocess_path, "wb") as f:
            pickle.dump((sliced_image, color_map), f)

    plot_sliced_image(color_map, sliced_image)
    plt.contour(np.flip(sliced_image.T, axis=0), colors="k")
    plt.show()
    print("stop here")
