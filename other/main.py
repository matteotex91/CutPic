from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

max_color = 255
zero_range = max_color / 6

CNX = 3
CNY = 3
CNZ = 3


def color_to_index(color, threshold):
    index = color / threshold + 1
    if index < 0:
        return 0
    elif index > 2:
        return 2
    else:
        return int(index)


def color_to_index_array(c_vec, threshold):
    return np.array(
        [
            color_to_index(c_vec[0], threshold),
            color_to_index(c_vec[1], threshold),
            color_to_index(c_vec[2], threshold),
        ]
    )


def project_colors(c_r, c_g, c_b, mat, threshold):
    count_mat = np.zeros((3, 3, 3))
    img_shape = np.shape(c_r)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            color_vec = np.array([c_r[i, j], c_g[i, j], c_b[i, j]])
            transf_color_vec = np.dot(mat, color_vec)
            indexes = color_to_index_array(transf_color_vec, threshold)
            count_mat[indexes[0], indexes[1], indexes[2]] += 1
    return np.std(count_mat)


if __name__ == "__main__":
    img = Image.open("picture4.jpg")
    rgb = img.load()
    img_shape = img.size
    pixels_count = img_shape[0] * img_shape[1]
    red = np.zeros(img_shape)
    green = np.zeros(img_shape)
    blue = np.zeros(img_shape)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            (red[i, j], green[i, j], blue[i, j]) = rgb[i, j]

    central_color = (np.average(red), np.average(green), np.average(blue))
    centered_red = red - central_color[0]
    centered_green = green - central_color[1]
    centered_blue = blue - central_color[2]

    in_ten = np.zeros((3, 3))
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            cv = (
                np.array(
                    [centered_red[i, j], centered_green[i, j], centered_blue[i, j]]
                )
                .repeat(3)
                .reshape((3, 3))
            )
            in_ten += cv * cv.T
    _, princ_axes = np.linalg.eig(in_ten)

    thresholds = np.linspace(0.01, max_color / 3, 25)

    best_threshold = thresholds[0]
    best_std = project_colors(
        centered_red, centered_green, centered_blue, princ_axes.T, best_threshold
    )
    std_queue = []

    for t in tqdm(thresholds):
        std = project_colors(
            centered_red, centered_green, centered_blue, princ_axes.T, t
        )
        if std < best_std:
            best_threshold = t
            best_std = std
        std_queue.append(std)

    discr_img = np.zeros(img_shape)
    color_count = np.zeros((27, 3))
    color_sum = np.zeros((27, 3))

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            color_vec = np.array(
                [centered_red[i, j], centered_green[i, j], centered_blue[i, j]]
            )
            transf_color_vec = np.dot(princ_axes.T, color_vec)
            indexes = color_to_index_array(transf_color_vec, best_threshold)
            lin_idx = indexes[0] * 9 + indexes[1] * 3 + indexes[2]
            color_count[lin_idx, :] += 1
            color_sum[lin_idx, :] += [
                red[i, j],
                green[i, j],
                blue[i, j],
            ]
            discr_img[i, j] = lin_idx
    discr_colors = color_sum / color_count / 256

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "custom cmap", discr_colors, np.shape(discr_colors)[0]
    )

    plt.pcolormesh(discr_img, cmap=cmap)
    plt.show()

    plt.contour(discr_img, colors="k")
    plt.show()

    fig, ax = plt.subplots(1, 2)
    ax[0].pcolormesh(np.flip(discr_img.T, axis=0), cmap=cmap)
    ax[1].contour(np.flip(discr_img.T, axis=0), colors="k")
    plt.show()

    # topological smoothening

    old_discr_img = np.copy(discr_img)

    topo_iter_num = 2
    for c in tqdm(
        [x for _, x in sorted(zip(color_count[:, 0], range(27)), reverse=True)]
    ):
        for k in range(topo_iter_num):
            new_img = np.copy(discr_img)
            for i in range(1, img_shape[0] - 1):
                for j in range(1, img_shape[1] - 1):
                    if discr_img[i, j] == c:
                        new_img[i - 1, j - 1] = c
                        new_img[i, j - 1] = c
                        new_img[i + 1, j - 1] = c
                        new_img[i - 1, j] = c
                        new_img[i + 1, j] = c
                        new_img[i - 1, j + 1] = c
                        new_img[i, j + 1] = c
                        new_img[i + 1, j + 1] = c
            discr_img = new_img

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].pcolormesh(old_discr_img, cmap=cmap)
    ax[0, 1].contour(old_discr_img, colors="k")
    ax[1, 0].pcolormesh(discr_img, cmap=cmap)
    ax[1, 1].contour(discr_img, colors="k")
    plt.show()
    print("stop here")
