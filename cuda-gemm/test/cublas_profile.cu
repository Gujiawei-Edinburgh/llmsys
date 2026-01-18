#include "../include/MatrixFP32.cuh"
#include "../include/utils.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Stats {
    float min_ms;
    float max_ms;
    float avg_ms;
};

static Stats compute_stats(const std::vector<float> &samples)
{
    Stats stats{0.0f, 0.0f, 0.0f};
    if (samples.empty()) {
        return stats;
    }
    stats.min_ms = *std::min_element(samples.begin(), samples.end());
    stats.max_ms = *std::max_element(samples.begin(), samples.end());
    float sum = 0.0f;
    for (float v : samples) {
        sum += v;
    }
    stats.avg_ms = sum / static_cast<float>(samples.size());
    return stats;
}

static float time_h2d(MatrixFP32 &h_mat, MatrixFP32 &d_mat, cudaEvent_t start, cudaEvent_t stop)
{
    cudaEventRecord(start);
    h_mat.copy_to_device(d_mat);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

static float time_d2h(MatrixFP32 &d_mat, MatrixFP32 &h_mat, cudaEvent_t start, cudaEvent_t stop)
{
    cudaEventRecord(start);
    d_mat.copy_to_host(h_mat);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

static float time_cublas_gemm(cublasHandle_t handle,
                              MatrixFP32 &d_a,
                              MatrixFP32 &d_b,
                              MatrixFP32 &d_c,
                              int rows,
                              int cols,
                              int kdim,
                              cudaEvent_t start,
                              cudaEvent_t stop)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaEventRecord(start);
    cublasCheck(cublasSgemm(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            cols,
                            rows,
                            kdim,
                            &alpha,
                            d_b.ptr,
                            cols,
                            d_a.ptr,
                            kdim,
                            &beta,
                            d_c.ptr,
                            cols));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main(int argc, char **argv)
{
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    int iters = 50;
    if (argc > 1) {
        sizes.clear();
        sizes.push_back(std::atoi(argv[1]));
    }
    if (argc > 2) {
        iters = std::max(1, std::atoi(argv[2]));
    }

    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    for (int size : sizes) {
        const int rows = size;
        const int cols = size;
        const int kdim = size;

        MatrixFP32 h_a(rows, kdim, false);
        MatrixFP32 h_b(kdim, cols, false);
        MatrixFP32 h_c(rows, cols, false);

        MatrixFP32 d_a(rows, kdim, true);
        MatrixFP32 d_b(kdim, cols, true);
        MatrixFP32 d_c(rows, cols, true);

        fill_random(h_a.ptr, rows * kdim);
        fill_random(h_b.ptr, kdim * cols);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warm up (context + kernels).
        h_a.copy_to_device(d_a);
        h_b.copy_to_device(d_b);
        time_cublas_gemm(handle, d_a, d_b, d_c, rows, cols, kdim, start, stop);
        d_c.copy_to_host(h_c);
        cudaDeviceSynchronize();

        std::vector<float> h2d_a_ms;
        std::vector<float> h2d_b_ms;
        std::vector<float> gemm_ms;
        std::vector<float> d2h_c_ms;
        h2d_a_ms.reserve(iters);
        h2d_b_ms.reserve(iters);
        gemm_ms.reserve(iters);
        d2h_c_ms.reserve(iters);

        for (int i = 0; i < iters; i++) {
            h2d_a_ms.push_back(time_h2d(h_a, d_a, start, stop));
            h2d_b_ms.push_back(time_h2d(h_b, d_b, start, stop));
            gemm_ms.push_back(time_cublas_gemm(handle, d_a, d_b, d_c, rows, cols, kdim, start, stop));
            d2h_c_ms.push_back(time_d2h(d_c, h_c, start, stop));
        }

        Stats h2d_a = compute_stats(h2d_a_ms);
        Stats h2d_b = compute_stats(h2d_b_ms);
        Stats gemm = compute_stats(gemm_ms);
        Stats d2h_c = compute_stats(d2h_c_ms);

        std::printf("N=%d iters=%d\n", size, iters);
        std::printf("  H2D A  avg=%.4f ms min=%.4f ms max=%.4f ms\n", h2d_a.avg_ms, h2d_a.min_ms, h2d_a.max_ms);
        std::printf("  H2D B  avg=%.4f ms min=%.4f ms max=%.4f ms\n", h2d_b.avg_ms, h2d_b.min_ms, h2d_b.max_ms);
        std::printf("  GEMM   avg=%.4f ms min=%.4f ms max=%.4f ms\n", gemm.avg_ms, gemm.min_ms, gemm.max_ms);
        std::printf("  D2H C  avg=%.4f ms min=%.4f ms max=%.4f ms\n", d2h_c.avg_ms, d2h_c.min_ms, d2h_c.max_ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        d_a.free_mat();
        d_b.free_mat();
        d_c.free_mat();
        h_a.free_mat();
        h_b.free_mat();
        h_c.free_mat();
    }

    cublasCheck(cublasDestroy(handle));
    return 0;
}
