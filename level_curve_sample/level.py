import numpy as np
import matplotlib.pyplot as plt


def scalar_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.exp(-np.power(x, 2) - np.power(y, 2))


def level_curve(x: np.ndarray, y: np.ndarray, z: np.ndarray, z0: float) -> np.ndarray:
    z_diff = z - z0
    z_shape = z.shape
    level_curve_points = np.empty(shape=(0, 4), dtype=float)

    for ix in range(z_shape[0] - 1):
        for iy in range(z_shape[1] - 1):
            y_xmin_cross = (z[ix, iy + 1] * y[ix, iy] - z[ix, iy] * y[ix, iy + 1]) / (
                y[ix, iy + 1] - y[ix, iy]
            )
            y_xmax_cross = (
                z[ix + 1, iy + 1] * y[ix + 1, iy] - z[ix + 1, iy] * y[ix + 1, iy + 1]
            ) / (y[ix + 1, iy + 1] - y[ix + 1, iy])

            x_ymin_cross = (z[ix + 1, iy] * x[ix, iy] - z[ix, iy] * x[ix + 1, iy]) / (
                x[ix + 1, iy] - x[ix + 1, iy]
            )
            x_ymax_cross = (
                z[ix + 1, iy + 1] * x[ix, iy + 1] - z[ix, iy + 1] * x[ix + 1, iy + 1]
            ) / (x[ix + 1, iy + 1] - x[ix, iy + 1])

    return level_curve_points


x_arr = np.linspace(-1, 1, 100)
y_arr = np.linspace(-1, 1, 100)
x_map, y_map = np.meshgrid(x_arr, y_arr)
z = scalar_function(x_map, y_map)
plt.pcolormesh(z)
plt.show()


print("stop here")
