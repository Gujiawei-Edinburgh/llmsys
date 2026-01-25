#define COARSE_FACTOR 8

#define TILE_A_ROWS 64
#define TILE_A_COLS 8
#define TILE_B_COLS 64

__global__ void coarse_1d_kernel(float *a, float *b, float*c, int C_rows, int C_cols, int A_cols)
{
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int A_view_ty = tx / tiles_A_cols; // map 2d to 1d view
    int A_view_tx = tx % tiles_A_cols;

    int B_view_ty = tx / tiles_B_cols;
    int B_view_tx = tx % tiles_B_cols;

    int row = TILE_A_ROWS*by + COARSE_FACTOR * (tx/TILE_B_COLS);
    int col = TILE_A_COLS*bx + (tx % tiles_B_cols);

    __shared__ float sh_a[TILE_A_ROWS][TILE_A_COLS];
    __shared__ float sh_b[TILE_A_COLS][TILE_B_COLS];

    int phases = ceil((float)A_cols/TILE_A_COLS);

    float coarsed_value[COARSE_FACTOR] = {0.0f};

    for (int p = 0; p < phases; p++)
    {
        if ((by * TILE_A_ROWS + A_view_ty < C_n_rows) && ((phase * TILE_A_COLS + A_view_tx) < A_cols))
        {
            sh_a[A_view_ty][A_view_tx] = a[(by * TILE_A_ROWS + A_view_ty) * A_cols + (phase*TILE_A_COLS+A_view_tx)];
        }
        else
        {
            sh_a[A_view_ty][A_view_tx] = 0.0f;
        }
        if ((phase * TILE_A_COLS + B_view_ty) < A_cols) && (bx * TILE_B_COLS + B_view_tx < C_cols))
        {
            sh_b[A_view_ty][A_view_tx] = b[(phase * TILE_A_COLS + B_view_ty)*C_cols + (bx * TILE_B_COLS + B_view_tx)];
        }
        else
        {
            sh_b[A_view_ty][A_view_tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_A_COLS; k++)
        {
            float reg = sh_b[k][B_view_tx];
            for (int c = 0; c < COARSE_FACTOR; c++)
            {
                coarsed_value[c] += sh_a[B_view_ty * COARSE_FACTOR+c][k] * reg;
            }
        }
        __syncthreads();
    }
}

void coarse_1d(float *a, float *b, float*c, int C_rows, int C_cols, int A_cols)
{
    dim3 block(TILE_A_ROWS * TILE_B_COLS / COARSE_FACTOR); // 1d block
    dim3 grid(ceil(C_cols/(float)(TILE_B_COLS)), ceil(C_rows/(float)(TILE_A_ROWS)));
    coarse_1d_kernel<<<grid, block>>>(a, b, c, C_rows, C_cols, A_cols);
}