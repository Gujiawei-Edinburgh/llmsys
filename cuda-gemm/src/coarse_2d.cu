# define COARSE_FACTOR_X 8
# define COARSE_FACTOR_Y 8

#define TILE_A_ROWS 128
#define TILE_A_COLS 16
#define TILE_B_COLS 128

__global__ void coarse_2d_kernel(float *a, float *b, float*c, int C_rows, int C_cols, int A_cols)
{
    int num_threads_per_blk = TILE_A_ROWS * TILE_B_COLS / (COARSE_FACTOR_X * COARSE_FACTOR_Y);
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int tx = threadIdx.x;

    int A_view_ty = tx / TILE_A_COLS;
    int A_view_tx = tx % TILE_A_COLS;
    int stride_A = num_threads_per_blk / TILE_A_COLS;
    int B_view_ty = tx / TILE_B_COLS;
    int B_view_tx = tx % TILE_B_COLS;
    int stride_B = num_threads_per_blk / TILE_B_COLS;

    // working on C[row][col]
    int row = COARSE_FACTOR_Y * (tx / (TILE_B_COLS/COARSE_FACTOR_X));
    int col = COARSE_FACTOR_X * (tx % (TILE_B_COLS/COARSE_FACTOR_X));

    __shared__ float sh_a[TILE_A_ROWS][TILE_A_COLS];
    __shared__ float sh_b[TILE_A_COLS][TILE_B_COLS];

    float coarsed_value[COARSE_FACTOR_Y][COARSE_FACTOR_X] = {0.0f};
    float register_A[COARSE_FACTOR_Y] = {0.0f};
    float register_B[COARSE_FACTOR_X] = {0.0f};

    int phases = ceil((float)A_cols/TILE_A_COLS);

    for (int phase = 0; phase < phases; phase++)
    {
        // load data to smem
        for (int offset = 0; offset < TILE_A_ROWS; offset += stride_A)
        {
            if ((by * TILE_A_ROWS + offset + A_view_ty < C_rows) && ((phase * TILE_A_COLS + A_view_tx) < A_cols))
            {
                sh_a[offset + A_view_ty][A_view_tx] = a[(by * TILE_A_ROWS + offset + A_view_ty) * A_cols + (phase * TILE_A_COLS + A_view_tx)];
            }
            else
            {
                sh_a[offset + A_view_ty][A_view_tx] = 0.0f;
            }
        }
        for (int offset = 0; offset < TILE_A_COLS; offset += stride_B)
        {
            if ((phase * TILE_A_COLS + offset + B_view_ty) < A_cols && (bx * TILE_B_COLS + B_view_tx < C_cols))
            {
                sh_b[offset + B_view_ty][B_view_tx] = b[(phase * TILE_A_COLS + offset + B_view_ty) * C_cols + (bx * TILE_B_COLS + B_view_tx)];
            }
            else
            {
                sh_b[offset + B_view_ty][B_view_tx] = 0.0f;
            }
        }
        __syncthreads();
        // calculation
        for (int k = 0; k < TILE_A_COLS; k++)
        {
            for (int i = 0; i < COARSE_FACTOR_Y; i++)
            {
                register_A[i] = sh_a[row + i][k];
            }
            for (int i = 0; i < COARSE_FACTOR_X; i++)
            {
                register_B[i] = sh_b[k][col + i];
            }
            for (int i = 0; i < COARSE_FACTOR_Y; i++)
            {
                for (int j = 0; j < COARSE_FACTOR_X; j++)
                {
                    coarsed_value[i][j] += register_A[i] * register_B[j];
                }
            }
        }
        __syncthreads();
    }

    for (int cy = 0; cy < COARSE_FACTOR_Y; cy++)
    {
        for (int cx = 0; cx < COARSE_FACTOR_X; cx++)
        {
            if ((by * TILE_A_ROWS + row + cy < C_rows) && (bx * TILE_B_COLS + col + cx < C_cols))
                c[(by * TILE_A_ROWS + row + cy) * C_cols + (bx * TILE_B_COLS + col + cx)] = coarsed_value[cy][cx];
        }
    }
}

void coarse_2d(float *a, float *b, float*c, int C_rows, int C_cols, int A_cols)
{
    dim3 block((TILE_A_ROWS * TILE_B_COLS) / (COARSE_FACTOR_X * COARSE_FACTOR_Y));
    dim3 grid(ceil(C_cols/(float)(TILE_B_COLS)), ceil(C_rows/(float)(TILE_A_ROWS)));
    coarse_2d_kernel<<<grid, block>>>(a, b, c, C_rows, C_cols, A_cols);
}