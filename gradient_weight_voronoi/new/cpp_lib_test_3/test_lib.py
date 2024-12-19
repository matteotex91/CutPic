import ctypes
import numpy as np
import os
from time import time
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
from tqdm import tqdm

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "voronoi.so"))
lib.voronoi.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int),
]

COLOR_RESOLUTION = 256
POINTS_COUNT = 25
R = (np.random.random(POINTS_COUNT) * COLOR_RESOLUTION).astype(np.int_)
G = (np.random.random(POINTS_COUNT) * COLOR_RESOLUTION).astype(np.int_)
B = (np.random.random(POINTS_COUNT) * COLOR_RESOLUTION).astype(np.int_)

wrapped_voronoi_map = np.zeros(
    (COLOR_RESOLUTION * COLOR_RESOLUTION * COLOR_RESOLUTION)
).astype(np.int_)


R_ptr = R.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
G_ptr = G.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
wrapped_voronoi_map_ptr = wrapped_voronoi_map.ctypes.data_as(
    ctypes.POINTER(ctypes.c_int)
)

t0 = time()
lib.voronoi(
    R_ptr, G_ptr, B_ptr, POINTS_COUNT, COLOR_RESOLUTION, wrapped_voronoi_map_ptr
)
print(f"lib processing time : {time()-t0}")

# unwrap the pointer
t0 = time()
# voronoi_map = np.zeros((COLOR_RESOLUTION, COLOR_RESOLUTION, COLOR_RESOLUTION))
# for ix in tqdm(range(COLOR_RESOLUTION)):
#     for iy in range(COLOR_RESOLUTION):
#         for iz in range(COLOR_RESOLUTION):
#             voronoi_map[ix, iy, iz] = wrapped_voronoi_map_ptr[
#                 ix + COLOR_RESOLUTION * (iy + COLOR_RESOLUTION * iz)
#             ]

voronoi_map = np.reshape(
    np.ravel(wrapped_voronoi_map),
    newshape=(COLOR_RESOLUTION, COLOR_RESOLUTION, COLOR_RESOLUTION),
)
print(f"unwrapping time : {time()-t0}")

print("stop here")

# interactive plot


fig, ax = plt.subplots()
B_index = 0


def on_release(event: KeyEvent) -> None:
    global B_index
    print(B_index)
    if event.key == "up":
        B_index = min(B_index + 1, COLOR_RESOLUTION)
    elif event.key == "down":
        B_index = max(B_index - 1, 0)
    redraw()


def redraw():
    global fig, ax, B_index
    plt.cla()
    ax.pcolormesh(voronoi_map[:, :, B_index])
    fig.show()


conn_id = fig.canvas.mpl_connect("key_release_event", on_release)
redraw()
plt.show()
fig.canvas.mpl_disconnect(conn_id)
print("stop here")
