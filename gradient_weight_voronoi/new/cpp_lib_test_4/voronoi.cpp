#include <cmath>
#include <cstddef>
#include <vector>

// compile command :
// g++ -shared -o voronoi.so -fPIC voronoi.cpp
// this method iteratively performs this procedure a specified number of times; if the specified condition is met, it stops:
// 1- start with a initial voronoi key vector set and calculate the relative voronoi tessellation
// 2- assign each pixel to the relative voronoi cell
// 3- compare the ration between the actual volume of each cell and the number of pixel belonging to it
// 4- compile che std of the ratio
// 5- if the relative std (std/avg) exceeds the threshold, go on. Otherwise stop and return
// 6- compute the c.o.m. of each subset of pixels belonging to each voronoi cell
// 7- use the c.o.m. set as the new set of voronoi key vectors and restart
//

extern "C" void voronoi(int *col_R,
                        int *col_G,
                        int *col_B,
                        int *col_vol,
                        const int *pix_R,
                        const int *pix_G,
                        const int *pix_B,
                        int *pix_vol,
                        int *wrapped_voronoi_map,
                        double *rel_std_sequence,
                        size_t pix_count,
                        size_t col_count,
                        size_t color_resolution,
                        double rel_std_threshold,
                        int n_iterations)
{
    int diff_r = 0;
    int diff_g = 0;
    int diff_b = 0;
    int closest_color_index = 0;
    int closest_color_norm = -1;
    int current_norm = 0;
    for (size_t iteration_index = 0; iteration_index < n_iterations; ++iteration_index)
    {
        for (size_t ix = 0; ix < color_resolution; ++ix)
        {
            for (size_t iy = 0; iy < color_resolution; ++iy)
            {
                for (size_t iz = 0; iz < color_resolution; ++iz)
                {
                    closest_color_norm = -1;
                    for (size_t color_index = 0; color_index < col_count; ++color_index)
                    {
                        diff_r = col_R[color_index] - ix;
                        diff_g = col_G[color_index] - iy;
                        diff_b = col_B[color_index] - iz;
                        current_norm = diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
                        if (closest_color_norm == -1)
                        {
                            closest_color_norm = current_norm;
                            closest_color_index = 0;
                        }
                        if (closest_color_norm > current_norm)
                        {
                            closest_color_norm = current_norm;
                            closest_color_index = color_index;
                        }
                    }
                    wrapped_voronoi_map[ix + color_resolution * (iy + color_resolution * iz)] = closest_color_index;
                    col_vol[closest_color_index]++;
                }
            }
        }
        for (size_t ip = 0; ip < pix_count; ++ip)
        {
            closest_color_norm = -1;
            for (size_t color_index = 0; color_index < col_count; ++color_index)
            {
                diff_r = col_R[color_index] - pix_R[ip];
                diff_g = col_G[color_index] - pix_G[ip];
                diff_b = col_B[color_index] - pix_B[ip];
                if (closest_color_norm == -1)
                {
                    closest_color_norm = current_norm;
                    closest_color_index = 0;
                }
                if (closest_color_norm > current_norm)
                {
                    closest_color_norm = current_norm;
                    closest_color_index = color_index;
                }
            }
            pix_vol[closest_color_index]++;
        }
        double ratio_count = 0;
        double ratio_sum = 0;
        double ratio_sum2 = 0;
        double ratio = 0;
        for (size_t color_index = 0; color_index < col_count; ++color_index)
        {
            ratio = pix_vol[color_index] / col_vol[color_index];
            ratio_count += 1;
            ratio_sum += ratio;
            ratio_sum2 += ratio * ratio;
        }
        double ratio_avg = ratio_sum / ratio_count;
        double ratio_std = std::sqrt(ratio_sum2 / ratio_count - (ratio_sum * ratio_sum) / (ratio_count * ratio_count));
        rel_std_sequence[iteration_index] = ratio_std / ratio_avg;
        if (ratio_std / ratio_avg < rel_std_threshold)
        {
            return;
        }
        }
}