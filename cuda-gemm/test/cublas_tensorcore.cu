#include "../include/MatrixFP32.cuh"
#include "../include/utils.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

void coarse_2d_tc(float *a, float *b, float *c, int C_rows, int C_cols, int A_cols);

int main()
{
    const int sizes[] = {1024, 2048, 4096, 8192};
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    for (int size : sizes) {
        const int rows = size;
        const int cols = size;
        const int kdim = size;

        MatrixFP32 h_a(rows, kdim, false);
        MatrixFP32 h_b(kdim, cols, false);
        MatrixFP32 h_c_naive(rows, cols, false);
        MatrixFP32 h_c_cublas(rows, cols, false);

        MatrixFP32 d_a(rows, kdim, true);
        MatrixFP32 d_b(kdim, cols, true);
        MatrixFP32 d_c_naive(rows, cols, true);
        MatrixFP32 d_c_cublas(rows, cols, true);

        fill_random(h_a.ptr, rows * kdim);
        fill_random(h_b.ptr, kdim * cols);

        h_a.copy_to_device(d_a);
        h_b.copy_to_device(d_b);

        // Warm up to reduce first-call overhead (context init, module load).
        coarse_2d_tc(d_a.ptr, d_b.ptr, d_c_naive.ptr, rows, cols, kdim);
        const float alpha = 1.0f;
        const float beta = 0.0f;
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
                                d_c_cublas.ptr,
                                cols));
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        coarse_2d_tc(d_a.ptr, d_b.ptr, d_c_naive.ptr, rows, cols, kdim);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float naive_ms = 0.0f;
        cudaEventElapsedTime(&naive_ms, start, stop);

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
                                d_c_cublas.ptr,
                                cols));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cublas_ms = 0.0f;
        cudaEventElapsedTime(&cublas_ms, start, stop);

        d_c_naive.copy_to_host(h_c_naive);
        d_c_cublas.copy_to_host(h_c_cublas);

        float max_abs_err = 0.0f;
        const int total = rows * cols;
        for (int i = 0; i < total; i++) {
            float err = std::fabs(h_c_naive.ptr[i] - h_c_cublas.ptr[i]);
            if (err > max_abs_err) {
                max_abs_err = err;
            }
        }

        const double flops = 2.0 * static_cast<double>(rows) * cols * kdim;
        const double naive_tflops = (flops / 1.0e12) / (naive_ms / 1.0e3);
        const double cublas_tflops = (flops / 1.0e12) / (cublas_ms / 1.0e3);

        std::printf("N=%d coarse_2d_tc=%.3f ms (%.3f TFLOP/s) cuBLAS=%.3f ms (%.3f TFLOP/s) max_abs_err=%.6f\n",
                    size,
                    naive_ms,
                    naive_tflops,
                    cublas_ms,
                    cublas_tflops,
                    max_abs_err);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        d_a.free_mat();
        d_b.free_mat();
        d_c_naive.free_mat();
        d_c_cublas.free_mat();
        h_a.free_mat();
        h_b.free_mat();
        h_c_naive.free_mat();
        h_c_cublas.free_mat();
    }

    cublasCheck(cublasDestroy(handle));
    return 0;
}