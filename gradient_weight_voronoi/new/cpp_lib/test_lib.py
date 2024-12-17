import ctypes
import numpy as np
import os

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "voronoi.so"))

lib.voronoi_norm.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.voronoi_norm.restype = ctypes.c_double

vec = np.random.rand(1000000).astype(np.float64)

vec_ptr = vec.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
length = len(vec)

norm = lib.voronoi_norm(vec_ptr, length)
print(f"La norma del vettore è: {norm}")
