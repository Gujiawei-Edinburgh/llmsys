#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>

namespace {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32; // 2-stage pipelining

static_assert((BM % WMMA_M) == 0, "BM must be multiple of 16");
static_assert((BN % WMMA_N) == 0, "BN must be multiple of 16");
static_assert((BK % WMMA_K) == 0, "BK must be multiple of 8");
static_assert((BK % 4) == 0, "BK must be multiple of 4 (float4 cp.async)");
static_assert((BN % 4) == 0, "BN must be multiple of 4 (float4 cp.async)");

__device__ __forceinline__ void cp_async_cg_16B(void *smem_dst, const void *gmem_src)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const std::uint32_t smem_addr = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(gmem_src));
#else
    (void)smem_dst;
    (void)gmem_src;
#endif
}

__device__ __forceinline__ void cp_async_commit()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" : :);
#endif
}

__device__ __forceinline__ void cp_async_wait_all()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 0;\n" : :);
#endif
}

template <int THREADS>
__device__ __forceinline__ void prefetch_tile_stage(const float *a,
                                                    const float *b,
                                                    float (*sh_a)[BM][BK],
                                                    float (*sh_b)[BK][BN],
                                                    int stage,
                                                    int block_m,
                                                    int block_n,
                                                    int k0,
                                                    int N,
                                                    int K)
{
    constexpr int A_VEC = (BM * BK) / 4; // float4 count
    constexpr int B_VEC = (BK * BN) / 4; // float4 count
    constexpr int TOTAL_VEC = A_VEC + B_VEC;

    const int tid = static_cast<int>(threadIdx.x);

    for (int vec = tid; vec < TOTAL_VEC; vec += THREADS) {
        if (vec < A_VEC) {
            const int row = vec / (BK / 4);
            const int k4 = vec % (BK / 4);
            const int k = k4 * 4;
            const float *g = a + (block_m + row) * K + (k0 + k);
            float *s = &sh_a[stage][row][k];
            cp_async_cg_16B(s, g);
        } else {
            const int v = vec - A_VEC;
            const int kk = v / (BN / 4);
            const int n4 = v % (BN / 4);
            const int n = n4 * 4;
            const float *g = b + (k0 + kk) * N + (block_n + n);
            float *s = &sh_b[stage][kk][n];
            cp_async_cg_16B(s, g);
        }
    }
}

__global__ void coarse_2d_tc_async_bn128_kernel(const float *a, const float *b, float *c, int M, int N, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    using namespace nvcuda::wmma;

    // 128x128 C tile per block:
    // - 8 warps cover M (one warp per 16 rows)
    // - 2 warps per M-strip cover N (each warp computes 16x64 = 4 WMMA tiles)
    constexpr int WARPS_PER_BLOCK = 16;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

    __shared__ __align__(16) float sh_a[2][BM][BK];
    __shared__ __align__(16) float sh_b[2][BK][BN];

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid / 32; // 0..15

    const int block_m = static_cast<int>(blockIdx.y) * BM;
    const int block_n = static_cast<int>(blockIdx.x) * BN;

    const int warp_m = warp_id / 2;         // 0..7
    const int warp_n_group = warp_id & 1;   // 0..1
    const int m0 = warp_m * WMMA_M;
    const int n_base = warp_n_group * 64;
    const int n0 = n_base + 0;
    const int n1 = n_base + 16;
    const int n2 = n_base + 32;
    const int n3 = n_base + 48;

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc0;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc1;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc2;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc3;
    fill_fragment(acc0, 0.0f);
    fill_fragment(acc1, 0.0f);
    fill_fragment(acc2, 0.0f);
    fill_fragment(acc3, 0.0f);

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> b_frag0;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> b_frag1;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> b_frag2;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> b_frag3;

    int stage = 0;
    prefetch_tile_stage<THREADS_PER_BLOCK>(a, b, sh_a, sh_b, stage, block_m, block_n, /*k0=*/0, N, K);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    for (int k0 = 0; k0 < K; k0 += BK) {
        const int next_k0 = k0 + BK;
        const int next_stage = stage ^ 1;
        if (next_k0 < K) {
            prefetch_tile_stage<THREADS_PER_BLOCK>(a, b, sh_a, sh_b, next_stage, block_m, block_n, next_k0, N, K);
            cp_async_commit();
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            load_matrix_sync(a_frag, &sh_a[stage][m0][kk], BK);
            load_matrix_sync(b_frag0, &sh_b[stage][kk][n0], BN);
            load_matrix_sync(b_frag1, &sh_b[stage][kk][n1], BN);
            load_matrix_sync(b_frag2, &sh_b[stage][kk][n2], BN);
            load_matrix_sync(b_frag3, &sh_b[stage][kk][n3], BN);
            mma_sync(acc0, a_frag, b_frag0, acc0);
            mma_sync(acc1, a_frag, b_frag1, acc1);
            mma_sync(acc2, a_frag, b_frag2, acc2);
            mma_sync(acc3, a_frag, b_frag3, acc3);
        }

        if (next_k0 < K) {
            cp_async_wait_all();
            __syncthreads();
            stage = next_stage;
        }
    }

    float *c0 = c + (block_m + m0) * N + (block_n + n0);
    float *c1 = c + (block_m + m0) * N + (block_n + n1);
    float *c2 = c + (block_m + m0) * N + (block_n + n2);
    float *c3 = c + (block_m + m0) * N + (block_n + n3);
    store_matrix_sync(c0, acc0, N, mem_row_major);
    store_matrix_sync(c1, acc1, N, mem_row_major);
    store_matrix_sync(c2, acc2, N, mem_row_major);
    store_matrix_sync(c3, acc3, N, mem_row_major);
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

// BN=128 version of the 2-stage cp.async TF32 Tensor Core kernel.
// Constraints (no boundary handling): M%128==0, N%128==0, K%32==0.
void coarse_2d_tc_async_bn128(float *a, float *b, float *c, int M, int N, int K)
{
    if ((M % BM) != 0 || (N % BN) != 0 || (K % BK) != 0) {
        return;
    }

    dim3 block(16 * 32);
    dim3 grid(N / BN, M / BM);
    coarse_2d_tc_async_bn128_kernel<<<grid, block>>>(a, b, c, M, N, K);
}

