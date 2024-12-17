import ctypes
import numpy as np
import os

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "voronoi.so"))
lib.vector_norms.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
]

vec = np.random.rand(1000000).astype(np.float64)
length = len(vec)

max_p = 5

norms = np.zeros(max_p, dtype=np.float64)

vec_ptr = vec.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
norms_ptr = norms.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

lib.vector_norms(vec_ptr, length, max_p, norms_ptr)

for p in range(1, max_p + 1):
    print(f"Norma L{p}: {norms[p-1]:.6f}")
