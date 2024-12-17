#include <cmath>
#include <cstddef>

// compile command :
// g++ -shared -o voronoi.so -fPIC voronoi.cpp

extern "C" double voronoi_norm(const double* vec, size_t length) {
    double sum = 0.0;
    for (size_t i = 0; i < length; ++i) {
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}

