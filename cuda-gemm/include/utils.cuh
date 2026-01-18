#ifndef UTILS_CUH
#define UTILS_CUH

#include <cublas_v2.h>
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

inline void cublasCheck(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(status));
        std::exit(1);
    }
}

inline void fill_random(float *buf, int count)
{
    for (int i = 0; i < count; i++) {
        buf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

#endif
