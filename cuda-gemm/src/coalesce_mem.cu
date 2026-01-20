#include "../include/MatrixFP32.cuh"

__global__ void coalesce_gemm_kernel(float *a, float *b, float *c, int C_rows, int C_cols, int A_cols)
{
    // working on C[row][col]
    // int row = blockIdx.x * blockDim.x + threadIdx.x;
    // int col = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    if (row < C_rows && col < C_cols) {
        float sum = 0.0f;
        for (int i = 0; i < A_cols; i++) {
            float A_val = a[row * A_cols + i];
            float B_val = b[i * C_cols + col];
            sum += A_val * B_val;
        }
        c[row * C_cols + col] = sum;
    }

}

void coalesce_gemm(float *a, float *b, float *c, int C_rows, int C_cols, int A_cols)
{
    dim3 block(16, 16);
    dim3 grid((C_rows + block.x - 1) / block.x, (C_cols + block.y - 1) / block.y);
    coalesce_gemm_kernel<<<grid, block>>>(a, b, c, C_rows, C_cols, A_cols);
}