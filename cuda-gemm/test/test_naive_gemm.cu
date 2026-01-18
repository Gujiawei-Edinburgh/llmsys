#include "../include/MatrixFP32.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

void naive_gemm(float *a, float *b, float *c, int C_rows, int C_cols, int A_cols);

static void cpu_gemm(const float *a, const float *b, float *c, int C_rows, int C_cols, int A_cols)
{
    for (int row = 0; row < C_rows; row++) {
        for (int col = 0; col < C_cols; col++) {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; k++) {
                sum += a[row * A_cols + k] * b[k * C_cols + col];
            }
            c[row * C_cols + col] = sum;
        }
    }
}

static void fill_random(float *buf, int count)
{
    for (int i = 0; i < count; i++) {
        buf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

int main()
{
    const int sizes[] = {128, 256, 512, 1024, 2048, 4096};

    for (int size : sizes) {
        const int rows = size;
        const int cols = size;
        const int kdim = size;

        MatrixFP32 h_a(rows, kdim, false);
        MatrixFP32 h_b(kdim, cols, false);
        MatrixFP32 h_c_cpu(rows, cols, false);
        MatrixFP32 h_c_gpu(rows, cols, false);

        MatrixFP32 d_a(rows, kdim, true);
        MatrixFP32 d_b(kdim, cols, true);
        MatrixFP32 d_c(rows, cols, true);

        fill_random(h_a.ptr, rows * kdim);
        fill_random(h_b.ptr, kdim * cols);

        h_a.copy_to_device(d_a);
        h_b.copy_to_device(d_b);

        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_gemm(h_a.ptr, h_b.ptr, h_c_cpu.ptr, rows, cols, kdim);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        naive_gemm(d_a.ptr, d_b.ptr, d_c.ptr, rows, cols, kdim);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, start, stop);

        d_c.copy_to_host(h_c_gpu);

        float max_abs_err = 0.0f;
        const int total = rows * cols;
        for (int i = 0; i < total; i++) {
            float err = std::fabs(h_c_cpu.ptr[i] - h_c_gpu.ptr[i]);
            if (err > max_abs_err) {
                max_abs_err = err;
            }
        }

        std::printf("N=%d CPU=%.3f ms GPU=%.3f ms max_abs_err=%.6f\n",
                    size, cpu_ms.count(), gpu_ms, max_abs_err);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        d_a.free_mat();
        d_b.free_mat();
        d_c.free_mat();
        h_a.free_mat();
        h_b.free_mat();
        h_c_cpu.free_mat();
        h_c_gpu.free_mat();
    }

    return 0;
}
