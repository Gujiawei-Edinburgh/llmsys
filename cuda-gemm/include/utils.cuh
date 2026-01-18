#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

inline void cudaCheck(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

#endif
