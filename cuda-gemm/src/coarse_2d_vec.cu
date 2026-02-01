# define COARSE_FACTOR_X 8
# define COARSE_FACTOR_Y 8

#define TILE_A_ROWS 128
#define TILE_A_COLS 16
#define TILE_B_COLS 128

__global__ void coarse_2d_vec_kernel(float *a, float *b, float*c, int C_rows, int C_cols, int A_cols)
{
    int num_threads_per_blk = TILE_A_ROWS * TILE_B_COLS / (COARSE_FACTOR_X * COARSE_FACTOR_Y);
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int tx = threadIdx.x;

    // Thread->tile views for vectorized (float4) loads.
    // A tile is [TILE_A_ROWS x TILE_A_COLS] = [128 x 16] floats = 512 float4s.
    // B tile is [TILE_A_COLS x TILE_B_COLS] = [16 x 128] floats = 512 float4s.
    constexpr int A_COLS4 = TILE_A_COLS / 4; // 4
    constexpr int B_COLS4 = TILE_B_COLS / 4; // 32

    int A_view_ty = tx / A_COLS4;
    int A_view_tx4 = tx % A_COLS4;
    int stride_A = num_threads_per_blk / A_COLS4; // rows per iteration

    int B_view_ty = tx / B_COLS4;
    int B_view_tx4 = tx % B_COLS4;
    int stride_B = num_threads_per_blk / B_COLS4; // k per iteration

    // working on C[row][col]
    int row = COARSE_FACTOR_Y * (tx / (TILE_B_COLS/COARSE_FACTOR_X));
    int col = COARSE_FACTOR_X * (tx % (TILE_B_COLS/COARSE_FACTOR_X));

    __shared__ float sh_a[TILE_A_ROWS][TILE_A_COLS];
    __shared__ float sh_b[TILE_A_COLS][TILE_B_COLS];

    float coarsed_value[COARSE_FACTOR_Y][COARSE_FACTOR_X] = {0.0f};
    float register_A[COARSE_FACTOR_Y] = {0.0f};
    float register_B[COARSE_FACTOR_X] = {0.0f};

    // Number of K-tiles (ceil division).
    int phases = (A_cols + TILE_A_COLS - 1) / TILE_A_COLS;

    for (int phase = 0; phase < phases; phase++)
    {
        // load data to smem
        for (int offset = 0; offset < TILE_A_ROWS; offset += stride_A)
        {
            int a_row = offset + A_view_ty;
            int g_row = by * TILE_A_ROWS + a_row;
            int g_col = phase * TILE_A_COLS + A_view_tx4 * 4;

            float4 tmp = make_float4(0.f, 0.f, 0.f, 0.f);
            if (g_row < C_rows && (g_col + 3) < A_cols)
            {
                tmp = *reinterpret_cast<const float4*>(&a[g_row * A_cols + g_col]);
            }
            else if (g_row < C_rows && g_col < A_cols)
            {
                // Tail (only hits if A_cols isn't a multiple of 4).
                tmp.x = a[g_row * A_cols + (g_col + 0)];
                tmp.y = (g_col + 1) < A_cols ? a[g_row * A_cols + (g_col + 1)] : 0.f;
                tmp.z = (g_col + 2) < A_cols ? a[g_row * A_cols + (g_col + 2)] : 0.f;
                tmp.w = (g_col + 3) < A_cols ? a[g_row * A_cols + (g_col + 3)] : 0.f;
            }

            if (a_row < TILE_A_ROWS)
            {
                sh_a[a_row][A_view_tx4 * 4 + 0] = tmp.x;
                sh_a[a_row][A_view_tx4 * 4 + 1] = tmp.y;
                sh_a[a_row][A_view_tx4 * 4 + 2] = tmp.z;
                sh_a[a_row][A_view_tx4 * 4 + 3] = tmp.w;
            }
        }
        for (int offset = 0; offset < TILE_A_COLS; offset += stride_B)
        {
            int b_row = offset + B_view_ty;
            int g_row = phase * TILE_A_COLS + b_row;
            int g_col = bx * TILE_B_COLS + B_view_tx4 * 4;

            float4 tmp = make_float4(0.f, 0.f, 0.f, 0.f);
            if (g_row < A_cols && (g_col + 3) < C_cols)
            {
                tmp = *reinterpret_cast<const float4*>(&b[g_row * C_cols + g_col]);
            }
            else if (g_row < A_cols && g_col < C_cols)
            {
                tmp.x = b[g_row * C_cols + (g_col + 0)];
                tmp.y = (g_col + 1) < C_cols ? b[g_row * C_cols + (g_col + 1)] : 0.f;
                tmp.z = (g_col + 2) < C_cols ? b[g_row * C_cols + (g_col + 2)] : 0.f;
                tmp.w = (g_col + 3) < C_cols ? b[g_row * C_cols + (g_col + 3)] : 0.f;
            }

            if (b_row < TILE_A_COLS)
            {
                sh_b[b_row][B_view_tx4 * 4 + 0] = tmp.x;
                sh_b[b_row][B_view_tx4 * 4 + 1] = tmp.y;
                sh_b[b_row][B_view_tx4 * 4 + 2] = tmp.z;
                sh_b[b_row][B_view_tx4 * 4 + 3] = tmp.w;
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

void coarse_2d_vec(float *a, float *b, float*c, int C_rows, int C_cols, int A_cols)
{
    dim3 block((TILE_A_ROWS * TILE_B_COLS) / (COARSE_FACTOR_X * COARSE_FACTOR_Y));
    dim3 grid(ceil(C_cols/(float)(TILE_B_COLS)), ceil(C_rows/(float)(TILE_A_ROWS)));
    coarse_2d_vec_kernel<<<grid, block>>>(a, b, c, C_rows, C_cols, A_cols);
}
