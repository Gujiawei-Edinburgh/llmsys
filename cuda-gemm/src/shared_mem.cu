#include "../include/MatrixFP32.cuh"
#include <assert.h>

#define TILE_WIDTH 32

__global__ void shared_mem_kernel(float *a, float *b, float*c, int C_rows, int C_cols, int A_cols) 
{
    __shared__ float sh_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_b[TILE_WIDTH][TILE_WIDTH];
    int phases = ceil((float)A_col/TILE_WIDTH)
}

void shared_mem(float *a, float *b, float*c, int C_rows, int C_cols, int A_cols) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((C_rows + block.x - 1) / block.x, (C_cols + block.y - 1) / block.y);
    shared_mem_kernel<<<grid, block>>>(a, b, c, C_rows, C_cols, A_cols);
}
