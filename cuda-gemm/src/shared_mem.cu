#include "../include/MatrixFP32.cuh"
#include <assert.h>

#define TILE_WIDTH 32

__global__ void shared_mem_kernel(float *a, float *b, float*c, int C_rows, int C_cols, int A_cols)
{
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);

    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = TILE_WIDTH*by + ty;
    int col = TILE_WIDTH*bx + tx;

    __shared__ float sh_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_b[TILE_WIDTH][TILE_WIDTH];
    int phases = ceil((float)A_cols/TILE_WIDTH);

    float sum = 0.0f;
    for (int p = 0; p < phases; p++)
    {
        int offset = p * TILE_WIDTH;
        if (row < C_rows && offset + tx < A_cols)
        {
            sh_a[ty][tx] = a[row * A_cols + offset + tx];
        }
        else
        {
            sh_a[ty][tx] = 0.0f;
        }

        if (col < C_cols && offset + ty < A_cols)
        {
            sh_b[ty][tx] = b[(offset + tx) * C_cols + col];
        }
        else
        {
            sh_b[ty][tx] = 0.0f;
        }

        __syncthreads(); // wait all shared mem loaded
        for (int k = 0; k < TILE_WIDTH; k++)
        {
            sum += sh_a[ty][k] * sh_b[k][tx];
        }

        __syncthreads(); // wait all cal done
    }
    if (row < C_rows && col < C_cols)
    {
        c[row * C_cols + col] = sum;
    }
}

void shared_mem(float *a, float *b, float*c, int C_rows, int C_cols, int A_cols) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((C_rows + block.x - 1) / block.x, (C_cols + block.y - 1) / block.y);
    shared_mem_kernel<<<grid, block>>>(a, b, c, C_rows, C_cols, A_cols);
}
