from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from scipy import stats


if __name__ == "__main__":
    img = Image.open("picture4.jpg")
    rgb = img.load()
    img_shape = img.size
    pixels_count = img_shape[0] * img_shape[1]
    red = np.zeros(img_shape)
    green = np.zeros(img_shape)
    blue = np.zeros(img_shape)
    hue = np.zeros(img_shape)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            (red[i, j], green[i, j], blue[i, j]) = rgb[i, j]
            hue[i, j] = np.arctan2(
                (-2 * red[i, j] + green[i, j] + blue[i, j]) / np.sqrt(3),
                (green[i, j] - blue[i, j]) / np.sqrt(2),
            )

    N = 25
    hue_div = 2 * np.pi / N

    quantiles = np.linspace(0, 1, N)
    qdiv = stats.mstats.mquantiles(hue.reshape((1, -1)), quantiles)

    # discr_img = np.zeros(img_shape)
    # for i in range(img_shape[0]):
    #     for j in range(img_shape[1]):
    #         discr_img[i, j] = int(hue[i, j] / hue_div)
    # discr_img -= np.min(discr_img)

    discr_img = np.zeros(img_shape)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            for q in range(N - 1):
                if hue[i, j] >= qdiv[q] and hue[i, j] < qdiv[q + 1]:
                    discr_img[i, j] = q

    color_count = np.zeros((N, 3))
    color_sum = np.zeros((N, 3))
    discr_img = np.int_(discr_img)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            color_count[discr_img[i, j], :] += 1
            color_sum[discr_img[i, j], :] += [red[i, j], green[i, j], blue[i, j]]

    discr_colors = color_sum / color_count / 256

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "custom cmap", discr_colors, np.shape(discr_colors)[0]
    )

    plt.pcolormesh(discr_img, cmap=cmap)
    plt.show()

    plt.contour(discr_img, colors="k")
    plt.show()

    print("stop here")
