#include <cmath>   // Per std::pow e std::sqrt
#include <cstddef> // Per size_t
#include <vector>  // Per std::vector

// compile command :
// g++ -shared -o voronoi.so -fPIC voronoi.cpp
// this method calculates L1,L2...LN norms of a nd vector. The N maximum degree of the norms is max_p parameter

extern "C" void vector_norms(const double* vec, size_t length, size_t max_p, double* norms) {
    for (size_t p = 1; p <= max_p; ++p) {
        double sum = 0.0;
        for (size_t i = 0; i < length; ++i) {
            sum += std::pow(std::abs(vec[i]), static_cast<double>(p));
        }
        norms[p - 1] = std::pow(sum, 1.0 / p); // Calcolo della norma Lp
    }
}
