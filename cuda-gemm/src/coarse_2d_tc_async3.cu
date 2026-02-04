#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

// Tile sizes: chosen so 3-stage pipelining fits under 99KB shared on SM89.
constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int STAGES = 3;

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

__device__ __forceinline__ void cp_async_wait_group(int n)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    switch (n) {
    case 0:
        asm volatile("cp.async.wait_group 0;\n" : :);
        break;
    case 1:
        asm volatile("cp.async.wait_group 1;\n" : :);
        break;
    default:
        asm volatile("cp.async.wait_group 2;\n" : :);
        break;
    }
#else
    (void)n;
#endif
}

__device__ __forceinline__ void prefetch_stage(const float *a,
                                               const float *b,
                                               float *sh_a,
                                               float *sh_b,
                                               int stage,
                                               int block_m,
                                               int block_n,
                                               int k0,
                                               int N,
                                               int K)
{
    // A: [BM x BK], B: [BK x BN], both row-major.
    constexpr int A_VEC = (BM * BK) / 4;
    constexpr int B_VEC = (BK * BN) / 4;
    constexpr int TOTAL_VEC = A_VEC + B_VEC;

    const int tid = static_cast<int>(threadIdx.x);
    constexpr int THREADS = 256;

    float *stage_a = sh_a + static_cast<std::size_t>(stage) * BM * BK;
    float *stage_b = sh_b + static_cast<std::size_t>(stage) * BK * BN;

    for (int vec = tid; vec < TOTAL_VEC; vec += THREADS) {
        if (vec < A_VEC) {
            const int row = vec / (BK / 4);
            const int k4 = vec % (BK / 4);
            const int k = k4 * 4;
            const float *g = a + (block_m + row) * K + (k0 + k);
            float *s = stage_a + row * BK + k;
            cp_async_cg_16B(s, g);
        } else {
            const int v = vec - A_VEC;
            const int kk = v / (BN / 4);
            const int n4 = v % (BN / 4);
            const int n = n4 * 4;
            const float *g = b + (k0 + kk) * N + (block_n + n);
            float *s = stage_b + kk * BN + n;
            cp_async_cg_16B(s, g);
        }
    }
}

__global__ void coarse_2d_tc_async3_kernel(const float *a, const float *b, float *c, int M, int N, int K)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    using namespace nvcuda::wmma;

    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    static_assert(THREADS_PER_BLOCK == 256, "expected 256 threads");

    extern __shared__ __align__(16) unsigned char smem_raw[];
    float *smem = reinterpret_cast<float *>(smem_raw);

    // Layout: [STAGES * BM*BK] then [STAGES * BK*BN]
    float *sh_a = smem;
    float *sh_b = smem + static_cast<std::size_t>(STAGES) * BM * BK;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid / 32; // 0..7

    const int block_m = static_cast<int>(blockIdx.y) * BM;
    const int block_n = static_cast<int>(blockIdx.x) * BN;

    const int m0 = warp_id * WMMA_M; // each warp covers a 16x64 strip
    constexpr int n0 = 0;
    constexpr int n1 = 16;
    constexpr int n2 = 32;
    constexpr int n3 = 48;

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

    const int num_tiles = K / BK;
    int issued = 0;
    const int initial = (num_tiles < STAGES) ? num_tiles : STAGES;

    #pragma unroll
    for (int s = 0; s < STAGES; s++) {
        if (s < initial) {
            prefetch_stage(a, b, sh_a, sh_b, s, block_m, block_n, /*k0=*/s * BK, N, K);
            cp_async_commit();
            issued++;
        }
    }

    int outstanding = issued;
    int tile_to_compute = 0;

    while (tile_to_compute < num_tiles) {
        // Make the oldest issued tile visible; keep (outstanding-1) groups in flight.
        cp_async_wait_group(outstanding - 1);
        __syncthreads();
        outstanding -= 1;

        const int stage = tile_to_compute % STAGES;
        float *stage_a = sh_a + static_cast<std::size_t>(stage) * BM * BK;
        float *stage_b = sh_b + static_cast<std::size_t>(stage) * BK * BN;

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            load_matrix_sync(a_frag, stage_a + (m0 * BK + kk), BK);
            load_matrix_sync(b_frag0, stage_b + (kk * BN + n0), BN);
            load_matrix_sync(b_frag1, stage_b + (kk * BN + n1), BN);
            load_matrix_sync(b_frag2, stage_b + (kk * BN + n2), BN);
            load_matrix_sync(b_frag3, stage_b + (kk * BN + n3), BN);
            mma_sync(acc0, a_frag, b_frag0, acc0);
            mma_sync(acc1, a_frag, b_frag1, acc1);
            mma_sync(acc2, a_frag, b_frag2, acc2);
            mma_sync(acc3, a_frag, b_frag3, acc3);
        }

        // Issue the next tile into the stage we just consumed.
        const int next_tile = tile_to_compute + STAGES;
        if (next_tile < num_tiles) {
            prefetch_stage(a, b, sh_a, sh_b, stage, block_m, block_n, /*k0=*/next_tile * BK, N, K);
            cp_async_commit();
            outstanding += 1;
        }

        tile_to_compute++;
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

void coarse_2d_tc_async3(float *a, float *b, float *c, int M, int N, int K)
{
    if ((M % BM) != 0 || (N % BN) != 0 || (K % BK) != 0) {
        return;
    }

    static bool configured = false;
    if (!configured) {
        const std::size_t bytes =
            static_cast<std::size_t>(STAGES) * (static_cast<std::size_t>(BM) * BK + static_cast<std::size_t>(BK) * BN) *
            sizeof(float);
        // Allow larger dynamic shared and prefer shared over L1 when possible.
        cudaFuncSetAttribute(coarse_2d_tc_async3_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(bytes));
        cudaFuncSetAttribute(coarse_2d_tc_async3_kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             100);
        configured = true;
    }

    const std::size_t smem_bytes =
        static_cast<std::size_t>(STAGES) * (static_cast<std::size_t>(BM) * BK + static_cast<std::size_t>(BK) * BN) *
        sizeof(float);

    dim3 block(256);
    dim3 grid(N / BN, M / BM);
    coarse_2d_tc_async3_kernel<<<grid, block, smem_bytes>>>(a, b, c, M, N, K);
}

