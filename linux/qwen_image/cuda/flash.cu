// Fused (flash) attention for the transformer on sm_75 FP16 tensor cores.
//
// out[q, h, :] = softmax(Q[q,h] . K[:,h]^T / sqrt(128), mask) . V[:, h] for
// head dim 128, without materializing the score matrix. Each warp owns 16
// query rows; a block of WARPS warps shares 64-key K/V tiles staged in shared
// memory. S = Q K^T and O += P V run on mma.m16n8k8 (FP16 in, FP32
// accumulation); the softmax is the online (running max / sum) form in FP32,
// in the log2 domain. P is reused from the S accumulator registers as the A
// operand of the second product, and V's B fragments come from ldmatrix.trans.
//
// Mask (as masked_softmax_kernel): global query row r = q + query_offset sees
// keys [0, visible_keys[r]) when visible_keys is given, else [0, r + 1) when
// r < causal_rows, else every key.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int HEAD_DIM = 128;
constexpr int TILE_KEYS = 64;
// Shared rows are padded by 8 halves so the 8 rows a fragment load touches
// fall in distinct banks.
constexpr int ROW = HEAD_DIM + 8;

__device__ __forceinline__ void mma_16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ uint32_t pack_half2(float low, float high) {
    const __half2 value = __floats2half2_rn(low, high);
    return *reinterpret_cast<const uint32_t *>(&value);
}

// Four transposed 8x8 fragments from shared memory (one address per lane).
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&out)[4], const __half *address) {
    const uint32_t shared = (uint32_t)__cvta_generic_to_shared(address);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(shared));
}

__device__ __forceinline__ int visible_for(int global_row, int keys, int causal_rows, const int *visible_keys) {
    if (visible_keys != nullptr) return visible_keys[global_row];
    return global_row < causal_rows ? global_row + 1 : keys;
}

template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
flash_attention_kernel(const float *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v,
                       float *__restrict__ out, int queries, int keys, int heads, int query_offset, int causal_rows,
                       const int *__restrict__ visible_keys) {
    constexpr int THREADS = WARPS * 32;
    constexpr int ROWS = WARPS * 16;
    // Vectors of 8 halves in one K (or V) tile, and per thread.
    constexpr int VECTORS = TILE_KEYS * HEAD_DIM / 8;
    constexpr int PER_THREAD = VECTORS / THREADS;
    __shared__ __align__(16) __half k_tile[TILE_KEYS * ROW];
    __shared__ __align__(16) __half v_tile[TILE_KEYS * ROW];
    __shared__ int block_visible;

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = lane >> 2, t = lane & 3;
    const int head = blockIdx.y;
    const int width = heads * HEAD_DIM;
    const int row0 = blockIdx.x * ROWS + warp * 16 + g;  // this thread's rows: row0, row0 + 8
    const int row1 = row0 + 8;

    // How far this block must read: the largest visible count among its rows.
    if (threadIdx.x == 0) block_visible = 0;
    __syncthreads();
    {
        const int r = blockIdx.x * ROWS + threadIdx.x % ROWS;
        if (threadIdx.x < ROWS && r < queries)
            atomicMax(&block_visible, visible_for(r + query_offset, keys, causal_rows, visible_keys));
    }
    __syncthreads();
    const int limit = block_visible < keys ? block_visible : keys;
    const int visible0 = row0 < queries ? visible_for(row0 + query_offset, keys, causal_rows, visible_keys) : 0;
    const int visible1 = row1 < queries ? visible_for(row1 + query_offset, keys, causal_rows, visible_keys) : 0;

    // Q fragments, pre-scaled so scores come out in the log2 domain.
    const float scale = 1.4426950408889634f / sqrtf((float)HEAD_DIM);
    uint32_t q_frag[16][2];
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
        const int column = head * HEAD_DIM + kk * 8 + t * 2;
        float2 a = make_float2(0.0f, 0.0f), b = make_float2(0.0f, 0.0f);
        if (row0 < queries) a = *reinterpret_cast<const float2 *>(q + (size_t)row0 * width + column);
        if (row1 < queries) b = *reinterpret_cast<const float2 *>(q + (size_t)row1 * width + column);
        q_frag[kk][0] = pack_half2(a.x * scale, a.y * scale);
        q_frag[kk][1] = pack_half2(b.x * scale, b.y * scale);
    }

    float o[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float max0 = -INFINITY, max1 = -INFINITY, sum0 = 0.0f, sum1 = 0.0f;

    // Register staging for the next tile: K and V vectors of 8 halves.
    uint4 k_next[PER_THREAD], v_next[PER_THREAD];
    auto load_tile = [&](int start) {
#pragma unroll
        for (int i = 0; i < PER_THREAD; ++i) {
            const int vector = threadIdx.x + i * THREADS;
            const int key = vector / (HEAD_DIM / 8), chunk = vector % (HEAD_DIM / 8);
            const int global_key = start + key;
            if (global_key < keys) {
                const size_t offset = (size_t)global_key * width + head * HEAD_DIM + chunk * 8;
                k_next[i] = *reinterpret_cast<const uint4 *>(k + offset);
                v_next[i] = *reinterpret_cast<const uint4 *>(v + offset);
            } else {
                k_next[i] = make_uint4(0, 0, 0, 0);
                v_next[i] = make_uint4(0, 0, 0, 0);
            }
        }
    };
    auto store_tile = [&]() {
#pragma unroll
        for (int i = 0; i < PER_THREAD; ++i) {
            const int vector = threadIdx.x + i * THREADS;
            const int key = vector / (HEAD_DIM / 8), chunk = vector % (HEAD_DIM / 8);
            *reinterpret_cast<uint4 *>(k_tile + key * ROW + chunk * 8) = k_next[i];
            *reinterpret_cast<uint4 *>(v_tile + key * ROW + chunk * 8) = v_next[i];
        }
    };

    const int tiles = (limit + TILE_KEYS - 1) / TILE_KEYS;
    if (tiles > 0) load_tile(0);
    for (int tile = 0; tile < tiles; ++tile) {
        const int start = tile * TILE_KEYS;
        __syncthreads();  // the previous tile's readers are done
        store_tile();
        __syncthreads();
        if (tile + 1 < tiles) load_tile(start + TILE_KEYS);

        // S = Q K^T for 16 rows x 64 keys: 8 n-tiles of 8 keys.
        float s[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            const __half *k_row = k_tile + (n * 8 + g) * ROW + t * 2;
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) {
                const uint32_t b0 = *reinterpret_cast<const uint32_t *>(k_row + kk * 8);
                mma_16816(s[n], q_frag[kk][0], q_frag[kk][1], b0);
            }
        }
        // Mask, then the online softmax update for rows row0 (s[.][0..1]) and
        // row1 (s[.][2..3]); a row's values are spread over the 4 lanes of a quad.
        float tile_max0 = -INFINITY, tile_max1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const int key = start + n * 8 + t * 2;
            if (key >= visible0) s[n][0] = -INFINITY;
            if (key + 1 >= visible0) s[n][1] = -INFINITY;
            if (key >= visible1) s[n][2] = -INFINITY;
            if (key + 1 >= visible1) s[n][3] = -INFINITY;
            tile_max0 = fmaxf(tile_max0, fmaxf(s[n][0], s[n][1]));
            tile_max1 = fmaxf(tile_max1, fmaxf(s[n][2], s[n][3]));
        }
#pragma unroll
        for (int o_ = 1; o_ < 4; o_ *= 2) {
            tile_max0 = fmaxf(tile_max0, __shfl_xor_sync(0xffffffffu, tile_max0, o_));
            tile_max1 = fmaxf(tile_max1, __shfl_xor_sync(0xffffffffu, tile_max1, o_));
        }
        const float new_max0 = fmaxf(max0, tile_max0), new_max1 = fmaxf(max1, tile_max1);
        // A row with nothing visible yet keeps a zero offset (all its p are 0).
        const float base0 = new_max0 == -INFINITY ? 0.0f : new_max0;
        const float base1 = new_max1 == -INFINITY ? 0.0f : new_max1;
        const float alpha0 = exp2f(max0 - base0), alpha1 = exp2f(max1 - base1);
        max0 = new_max0;
        max1 = new_max1;
        float tile_sum0 = 0.0f, tile_sum1 = 0.0f;
        uint32_t p[8][2];
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const float p0 = exp2f(s[n][0] - base0), p1 = exp2f(s[n][1] - base0);
            const float p2 = exp2f(s[n][2] - base1), p3 = exp2f(s[n][3] - base1);
            tile_sum0 += p0 + p1;
            tile_sum1 += p2 + p3;
            p[n][0] = pack_half2(p0, p1);
            p[n][1] = pack_half2(p2, p3);
        }
        sum0 = sum0 * alpha0 + tile_sum0;
        sum1 = sum1 * alpha1 + tile_sum1;
#pragma unroll
        for (int n = 0; n < 16; ++n) {
            o[n][0] *= alpha0;
            o[n][1] *= alpha0;
            o[n][2] *= alpha1;
            o[n][3] *= alpha1;
        }
        // O += P V: 8 key steps x 16 n-tiles of 8 dims; B fragments via
        // ldmatrix.trans, four n-tiles per load. Lane l addresses row l % 8 of
        // matrix l / 8.
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int n4 = 0; n4 < 16; n4 += 4) {
                uint32_t b[4];
                ldmatrix_x4_trans(b, v_tile + (j * 8 + (lane & 7)) * ROW + (n4 + (lane >> 3)) * 8);
#pragma unroll
                for (int i = 0; i < 4; ++i) mma_16816(o[n4 + i], p[j][0], p[j][1], b[i]);
            }
        }
    }

    // Finish the sums across the quad and write O / sum.
#pragma unroll
    for (int o_ = 1; o_ < 4; o_ *= 2) {
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, o_);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, o_);
    }
    const float inverse0 = sum0 > 0.0f ? 1.0f / sum0 : 0.0f;
    const float inverse1 = sum1 > 0.0f ? 1.0f / sum1 : 0.0f;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
        const int column = head * HEAD_DIM + n * 8 + t * 2;
        if (row0 < queries)
            *reinterpret_cast<float2 *>(out + (size_t)row0 * width + column) =
                make_float2(o[n][0] * inverse0, o[n][1] * inverse0);
        if (row1 < queries)
            *reinterpret_cast<float2 *>(out + (size_t)row1 * width + column) =
                make_float2(o[n][2] * inverse1, o[n][3] * inverse1);
    }
}

}  // namespace

// Flash attention with FP16 K and V ([keys, heads * 128]) and FP32 Q and out
// ([queries, heads * 128]); see qi_attention_half for the mask arguments.
extern "C" int qi_flash_attention(const float *q, const void *k_half, const void *v_half, float *out, int queries, int keys,
                                  int heads, int query_offset, int causal_rows, const int *visible_keys) {
    constexpr int WARPS = 8;
    const dim3 grid((queries + WARPS * 16 - 1) / (WARPS * 16), heads);
    flash_attention_kernel<WARPS><<<grid, WARPS * 32>>>(q, (const __half *)k_half, (const __half *)v_half, out, queries, keys,
                                                        heads, query_offset, causal_rows, visible_keys);
    return (int)cudaGetLastError();
}
