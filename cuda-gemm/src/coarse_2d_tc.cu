#include <cuda_runtime.h>
#include <mma.h>

namespace {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32; // must be multiple of WMMA_K

static_assert(BK % WMMA_K == 0, "BK must be a multiple of WMMA_K");
static_assert(BM % WMMA_M == 0, "BM must be a multiple of WMMA_M");
static_assert(BN % WMMA_N == 0, "BN must be a multiple of WMMA_N");

__global__ void coarse_2d_tc_kernel(const float *a,
                                        const float *b,
                                        float *c,
                                        int M,
                                        int N,
                                        int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    using namespace nvcuda::wmma;

    constexpr int WARPS_PER_BLOCK = 8; // 256 threads
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

    __shared__ __align__(16) float sh_a[BM][BK];
    __shared__ __align__(16) float sh_b[BN][BK]; // transposed: [n][k] so it is col-major with ld=BK

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid / 32;

    const int block_m = static_cast<int>(blockIdx.y) * BM;
    const int block_n = static_cast<int>(blockIdx.x) * BN;

    const int warp_m = warp_id / 2;               // 0..3
    const int warp_n_base = (warp_id % 2) * 32;   // 0 or 32
    const int m0 = warp_m * WMMA_M;
    const int n0 = warp_n_base;
    const int n1 = warp_n_base + WMMA_N;

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc0;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc1;
    fill_fragment(acc0, 0.0f);
    fill_fragment(acc1, 0.0f);

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, col_major> b_frag0;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, col_major> b_frag1;

    for (int k0 = 0; k0 < K; k0 += BK) {
        const int a_elems = BM * BK;
        const int b_elems = BN * BK;
        const int total = a_elems + b_elems;

        for (int idx = tid; idx < total; idx += THREADS_PER_BLOCK) {
            if (idx < a_elems) {
                const int m = idx / BK;
                const int k = idx % BK;
                sh_a[m][k] = a[(block_m + m) * K + (k0 + k)];
            } else {
                const int j = idx - a_elems;
                const int n = j / BK;
                const int k = j % BK;
                sh_b[n][k] = b[(k0 + k) * N + (block_n + n)];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            load_matrix_sync(a_frag, &sh_a[m0][kk], BK);
            load_matrix_sync(b_frag0, &sh_b[n0][kk], BK);
            load_matrix_sync(b_frag1, &sh_b[n1][kk], BK);
            mma_sync(acc0, a_frag, b_frag0, acc0);
            mma_sync(acc1, a_frag, b_frag1, acc1);
        }
        __syncthreads();
    }

    float *c0 = c + (block_m + m0) * N + (block_n + n0);
    float *c1 = c + (block_m + m0) * N + (block_n + n1);
    store_matrix_sync(c0, acc0, N, mem_row_major);
    store_matrix_sync(c1, acc1, N, mem_row_major);
#else
    (void)a;
    (void)b;
    (void)c;
    (void)M;
    (void)N;
    (void)K;
#endif
}

} // namespace

bool coarse_2d_tc(const float *a, const float *b, float *c, int M, int N, int K)
{
    if ((M % BM) != 0 || (N % BN) != 0 || (K % BK) != 0) {
        return false;
    }

    dim3 block(256);
    dim3 grid(N / BN, M / BM);
    coarse_2d_tf32_tc_kernel<<<grid, block>>>(a, b, c, M, N, K);
    return true;
}
