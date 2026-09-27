// VAE decoder operations (AutoencoderKLQwenImage21, one frame). Activations
// are FP32 NCHW with batch 1. Convolutions run through cuDNN in FP32.
//
// The decoder's 3D pieces reduce to 2D for the first (only) frame: causal 3x3
// convolutions pad spatially only, `upsample3d` skips its time_conv on the first
// chunk, and the DupUp3D shortcut keeps its last temporal slice (see
// dup_up_add_kernel).

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cmath>

static cudnnHandle_t cudnn = nullptr;
static cublasHandle_t vae_blas = nullptr;
static void *workspace = nullptr;
static size_t workspace_bytes = 0;

static unsigned blocks_for(unsigned long long count, unsigned threads) {
    return (unsigned)((count + threads - 1) / threads);
}

__global__ void add_bias_kernel(float *y, const float *bias, int channels, int pixels) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < (unsigned long long)channels * pixels) y[i] += bias[i / pixels];
}

// MARK: - FP16 tensor-core convolution (stride 1)

// Stride-1 convolutions on FP16 tensor cores: cuDNN is fast here only for FP16
// NHWC (implicit precomputed GEMM), about 2x FP32 Winograd. Activations stay
// FP32 NCHW in memory; each horizontal strip of output rows converts its input
// rows plus a k/2 halo (zeros outside the image) to FP16 NHWC, runs with
// vertical padding 0 and horizontal padding k/2, and converts back with the
// bias. Strips keep the FP16 scratch bounded at large sizes.
static void *half_input = nullptr, *half_output = nullptr, *half_filter = nullptr;
static size_t half_input_bytes = 0, half_output_bytes = 0, half_filter_bytes = 0;
static const size_t HALF_STRIP_BYTES = (size_t)96 << 20;

static int ensure(void **buffer, size_t *capacity, size_t bytes) {
    if (bytes <= *capacity) return 0;
    cudaFree(*buffer);
    *buffer = nullptr;
    *capacity = 0;
    if (cudaMalloc(buffer, bytes) != cudaSuccess) return (int)cudaErrorMemoryAllocation;
    *capacity = bytes;
    return 0;
}

// Within a strip both layouts are a matrix: NCHW is [C][positions] (a
// channel's rows are contiguous) and NHWC is [positions][C]. The conversions
// are 32 x 32 tiled transposes through shared memory, coalesced both ways.

// Rows [row0 - pad, row0 - pad + rows_in) of x [C, H, W] into NHWC halves,
// zeros for rows outside the image.
__global__ void strip_to_nhwc_kernel(const float *x, __half *out, int channels, int height, int width, int row0, int rows_in,
                                     int pad) {
    __shared__ float tile[32][33];
    const long long positions = (long long)rows_in * width;
    const long long first = (long long)(row0 - pad) * width;  // image offset of position 0
    const long long image = (long long)height * width;
    const long long p0 = (long long)blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    for (int j = threadIdx.y; j < 32; j += 8) {
        const int c = c0 + j;
        const long long p = p0 + threadIdx.x, at = first + p;
        tile[j][threadIdx.x] = c < channels && p < positions && at >= 0 && at < image ? x[(size_t)c * image + at] : 0.0f;
    }
    __syncthreads();
    for (int j = threadIdx.y; j < 32; j += 8) {
        const long long p = p0 + j;
        const int c = c0 + threadIdx.x;
        if (p < positions && c < channels) out[(size_t)p * channels + c] = __float2half(tile[threadIdx.x][j]);
    }
}

// NHWC halves [rows, W, C] back to rows [row0, row0 + rows) of y [C, H, W], plus bias.
__global__ void strip_from_nhwc_kernel(const __half *in, const float *bias, float *y, int channels, int height, int width,
                                       int row0, int rows) {
    __shared__ float tile[32][33];
    const long long positions = (long long)rows * width;
    const long long first = (long long)row0 * width;
    const long long image = (long long)height * width;
    const long long p0 = (long long)blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    for (int j = threadIdx.y; j < 32; j += 8) {
        const long long p = p0 + j;
        const int c = c0 + threadIdx.x;
        tile[j][threadIdx.x] = p < positions && c < channels ? __half2float(in[(size_t)p * channels + c]) : 0.0f;
    }
    __syncthreads();
    for (int j = threadIdx.y; j < 32; j += 8) {
        const int c = c0 + j;
        const long long p = p0 + threadIdx.x;
        if (c < channels && p < positions) y[(size_t)c * image + first + p] = tile[threadIdx.x][j] + bias[c];
    }
}

// w [K, C, R, S] FP32 to [K, R, S, C] halves (cuDNN's NHWC filter layout).
__global__ void filter_to_krsc_kernel(const float *w, __half *out, int k_count, int c_count, int kernel) {
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    const unsigned long long count = (unsigned long long)k_count * c_count * kernel * kernel;
    if (i >= count) return;
    const int c = (int)(i % c_count);
    const int s = (int)((i / c_count) % kernel);
    const int r = (int)((i / ((unsigned long long)c_count * kernel)) % kernel);
    const int k = (int)(i / ((unsigned long long)c_count * kernel * kernel));
    out[i] = __float2half(w[(((size_t)k * c_count + c) * kernel + r) * kernel + s]);
}

static int conv2d_half(const float *x, const float *weights, const float *bias, float *y, int in_channels,
                       int out_channels, int height, int width, int kernel) {
    const int pad = kernel / 2;
    const size_t widest = (size_t)(in_channels > out_channels ? in_channels : out_channels);
    int rows = (int)(HALF_STRIP_BYTES / (widest * width * 2)) - 2 * pad;
    if (rows < 1) rows = 1;
    if (rows > height) rows = height;
    const size_t filter_count = (size_t)out_channels * in_channels * kernel * kernel;
    if (int status = ensure(&half_input, &half_input_bytes, (size_t)(rows + 2 * pad) * width * in_channels * 2)) return status;
    if (int status = ensure(&half_output, &half_output_bytes, (size_t)rows * width * out_channels * 2)) return status;
    if (int status = ensure(&half_filter, &half_filter_bytes, filter_count * 2)) return status;
    filter_to_krsc_kernel<<<blocks_for(filter_count, 256), 256>>>(weights, (__half *)half_filter, out_channels, in_channels, kernel);
    cudnnTensorDescriptor_t input, output;
    cudnnFilterDescriptor_t filter;
    cudnnConvolutionDescriptor_t convolution;
    cudnnCreateTensorDescriptor(&input);
    cudnnCreateTensorDescriptor(&output);
    cudnnCreateFilterDescriptor(&filter);
    cudnnCreateConvolutionDescriptor(&convolution);
    int status = 0;
    if (cudnnSetFilter4dDescriptor(filter, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, out_channels, in_channels, kernel, kernel) ||
        cudnnSetConvolution2dDescriptor(convolution, 0, pad, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) ||
        cudnnSetConvolutionMathType(convolution, CUDNN_TENSOR_OP_MATH)) {
        status = (int)cudaErrorInvalidValue;
    }
    const cudnnConvolutionFwdAlgo_t algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    const float one = 1.0f, zero = 0.0f;
    for (int row0 = 0; status == 0 && row0 < height; row0 += rows) {
        const int strip = height - row0 < rows ? height - row0 : rows;
        const int rows_in = strip + 2 * pad;
        if (cudnnSetTensor4dDescriptor(input, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 1, in_channels, rows_in, width) ||
            cudnnSetTensor4dDescriptor(output, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 1, out_channels, strip, width)) {
            status = (int)cudaErrorInvalidValue;
            break;
        }
        size_t needed = 0;
        if (cudnnGetConvolutionForwardWorkspaceSize(cudnn, input, filter, convolution, output, algorithm, &needed) != CUDNN_STATUS_SUCCESS) {
            status = (int)cudaErrorUnknown;
            break;
        }
        if (needed > workspace_bytes) {
            cudaFree(workspace);
            workspace = nullptr;
            workspace_bytes = 0;
            if (cudaMalloc(&workspace, needed) != cudaSuccess) { status = (int)cudaErrorMemoryAllocation; break; }
            workspace_bytes = needed;
        }
        const dim3 tiles_in((unsigned)(((long long)rows_in * width + 31) / 32), (unsigned)((in_channels + 31) / 32));
        strip_to_nhwc_kernel<<<tiles_in, dim3(32, 8)>>>(x, (__half *)half_input, in_channels, height, width, row0, rows_in, pad);
        if (cudnnConvolutionForward(cudnn, &one, input, half_input, filter, half_filter, convolution, algorithm, workspace,
                                    workspace_bytes, &zero, output, half_output) != CUDNN_STATUS_SUCCESS) {
            status = (int)cudaErrorUnknown;
            break;
        }
        const dim3 tiles_out((unsigned)(((long long)strip * width + 31) / 32), (unsigned)((out_channels + 31) / 32));
        strip_from_nhwc_kernel<<<tiles_out, dim3(32, 8)>>>((const __half *)half_output, bias, y, out_channels, height, width, row0,
                                                           strip);
    }
    cudnnDestroyConvolutionDescriptor(convolution);
    cudnnDestroyFilterDescriptor(filter);
    cudnnDestroyTensorDescriptor(output);
    cudnnDestroyTensorDescriptor(input);
    if (status != 0) return status;
    return (int)cudaGetLastError();
}

// y[out, H', W'] = conv(x[in, H, W], w[out, in, k, k]) + bias with symmetric
// zero padding `pad` and stride `stride`; H' = (H + 2 pad - k) / stride + 1.
extern "C" int qi_vae_conv2d_strided(const float *x, const float *weights, const float *bias, float *y,
                                     int in_channels, int out_channels, int height, int width, int kernel,
                                     int stride, int pad) {
    if (cudnn == nullptr && cudnnCreate(&cudnn) != CUDNN_STATUS_SUCCESS) return (int)cudaErrorInitializationError;
    cudnnTensorDescriptor_t input, output;
    cudnnFilterDescriptor_t filter;
    cudnnConvolutionDescriptor_t convolution;
    cudnnCreateTensorDescriptor(&input);
    cudnnCreateTensorDescriptor(&output);
    cudnnCreateFilterDescriptor(&filter);
    cudnnCreateConvolutionDescriptor(&convolution);
    int status = 0;
    const int out_height = (height + 2 * pad - kernel) / stride + 1;
    const int out_width = (width + 2 * pad - kernel) / stride + 1;
    if (cudnnSetTensor4dDescriptor(input, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, in_channels, height, width) ||
        cudnnSetTensor4dDescriptor(output, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, out_height, out_width) ||
        cudnnSetFilter4dDescriptor(filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels, kernel, kernel) ||
        cudnnSetConvolution2dDescriptor(convolution, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT)) {
        status = (int)cudaErrorInvalidValue;
    }
    cudnnConvolutionFwdAlgoPerf_t choices[8];
    int returned = 0;
    if (status == 0 && cudnnGetConvolutionForwardAlgorithm_v7(cudnn, input, filter, convolution, output, 8, &returned, choices)
                           != CUDNN_STATUS_SUCCESS) {
        status = (int)cudaErrorUnknown;
    }
    cudnnConvolutionFwdAlgo_t algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t needed = 0;
    for (int i = 0; status == 0 && i < returned; ++i) {
        if (choices[i].status != CUDNN_STATUS_SUCCESS) continue;
        // Keep workspace bounded; implicit GEMM needs none.
        if (choices[i].memory > (size_t)512 << 20) continue;
        algorithm = choices[i].algo;
        needed = choices[i].memory;
        break;
    }
    if (status == 0 && needed > workspace_bytes) {
        cudaFree(workspace);
        workspace = nullptr;
        workspace_bytes = 0;
        if (cudaMalloc(&workspace, needed) != cudaSuccess) status = (int)cudaErrorMemoryAllocation;
        else workspace_bytes = needed;
    }
    const float one = 1.0f, zero = 0.0f;
    if (status == 0 && cudnnConvolutionForward(cudnn, &one, input, x, filter, weights, convolution, algorithm,
                                               workspace, workspace_bytes, &zero, output, y) != CUDNN_STATUS_SUCCESS) {
        status = (int)cudaErrorUnknown;
    }
    cudnnDestroyConvolutionDescriptor(convolution);
    cudnnDestroyFilterDescriptor(filter);
    cudnnDestroyTensorDescriptor(output);
    cudnnDestroyTensorDescriptor(input);
    if (status != 0) return status;
    const int pixels = out_height * out_width;
    add_bias_kernel<<<blocks_for((unsigned long long)out_channels * pixels, 256), 256>>>(y, bias, out_channels, pixels);
    return (int)cudaGetLastError();
}

// Same-size convolution: stride 1, zero padding k/2, on FP16 tensor cores.
extern "C" int qi_vae_conv2d(const float *x, const float *weights, const float *bias, float *y,
                             int in_channels, int out_channels, int height, int width, int kernel) {
    if (cudnn == nullptr && cudnnCreate(&cudnn) != CUDNN_STATUS_SUCCESS) return (int)cudaErrorInitializationError;
    return conv2d_half(x, weights, bias, y, in_channels, out_channels, height, width, kernel);
}

// y[c, H + 1, W + 1] = x[c, H, W] with one zero row below and one zero column
// to the right (the encoder downsampler's ZeroPad2d((0, 1, 0, 1))).
__global__ void pad_bottom_right_kernel(const float *x, float *y, int height, int width, unsigned long long count) {
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int padded_width = width + 1;
    const int column = (int)(i % padded_width);
    const int row = (int)((i / padded_width) % (height + 1));
    const unsigned long long channel = i / ((unsigned long long)padded_width * (height + 1));
    y[i] = row < height && column < width ? x[(channel * height + row) * width + column] : 0.0f;
}

extern "C" int qi_vae_pad_bottom_right(const float *x, float *y, int channels, int height, int width) {
    const unsigned long long count = (unsigned long long)channels * (height + 1) * (width + 1);
    pad_bottom_right_kernel<<<blocks_for(count, 256), 256>>>(x, y, height, width, count);
    return (int)cudaGetLastError();
}

// Encoder AvgDown3D shortcut for a single frame, added into y[out, H/f, W/f].
// Channels, time and space fold as (c, t, sh, sw) into in * f_t * f_s^2 values
// that are averaged in consecutive groups; with f_t = 2 the one frame is
// front-padded, so the t = 0 entries are zeros.
__global__ void avg_down_add_kernel(float *y, const float *x, int in_channels, int out_channels, int height, int width,
                                    int factor_t, int factor_s) {
    const int out_height = height / factor_s, out_width = width / factor_s;
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= (unsigned long long)out_channels * out_height * out_width) return;
    const int w = (int)(i % out_width), h = (int)((i / out_width) % out_height), o = (int)(i / ((unsigned long long)out_width * out_height));
    const int factor = factor_t * factor_s * factor_s;
    const int group = in_channels * factor / out_channels;
    float sum = 0.0f;
    for (int g = 0; g < group; ++g) {
        const int flat = o * group + g;
        const int c = flat / factor, rem = flat % factor;
        const int t = rem / (factor_s * factor_s), sh = (rem / factor_s) % factor_s, sw = rem % factor_s;
        if (t == factor_t - 1) sum += x[((size_t)c * height + h * factor_s + sh) * width + w * factor_s + sw];
    }
    y[i] += sum / (float)group;
}

extern "C" int qi_vae_avg_down_add(float *y, const float *x, int in_channels, int out_channels, int height, int width,
                                   int factor_t, int factor_s) {
    const unsigned long long count = (unsigned long long)out_channels * (height / factor_s) * (width / factor_s);
    avg_down_add_kernel<<<blocks_for(count, 256), 256>>>(y, x, in_channels, out_channels, height, width, factor_t, factor_s);
    return (int)cudaGetLastError();
}

// quant_conv output [128, N] to packed, normalized condition latents [N, 64]:
// the posterior mean (first 64 channels), (z - mean) / std.
__global__ void pack_latents_kernel(const float *z, const float *mean, const float *std, float *latents, int pixels) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixels * 64) return;
    const int p = i / 64, c = i % 64;
    latents[i] = (z[(size_t)c * pixels + p] - mean[c]) / std[c];
}

extern "C" int qi_vae_pack_latents(const float *z, const float *mean, const float *std, float *latents, int pixels) {
    pack_latents_kernel<<<blocks_for((unsigned long long)pixels * 64, 256), 256>>>(z, mean, std, latents, pixels);
    return (int)cudaGetLastError();
}

// QwenImage21RMS_norm over channels at each pixel: x / max(||x||, 1e-12) *
// sqrt(C) * gamma[c], then SiLU when `silu` is set. One thread per pixel.
__global__ void channel_rms_norm_kernel(const float *x, const float *gamma, float *y, int channels, int pixels, int silu) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    float squares = 0.0f;
    for (int c = 0; c < channels; ++c) {
        const float v = x[(size_t)c * pixels + p];
        squares += v * v;
    }
    const float scale = sqrtf((float)channels) / fmaxf(sqrtf(squares), 1e-12f);
    for (int c = 0; c < channels; ++c) {
        float v = x[(size_t)c * pixels + p] * scale * gamma[c];
        if (silu) v = v / (1.0f + expf(-v));
        y[(size_t)c * pixels + p] = v;
    }
}

extern "C" int qi_vae_rms_norm(const float *x, const float *gamma, float *y, int channels, int pixels, int silu) {
    channel_rms_norm_kernel<<<blocks_for(pixels, 128), 128>>>(x, gamma, y, channels, pixels, silu);
    return (int)cudaGetLastError();
}

// Nearest 2x: y[c, 2h + a, 2w + b] = x[c, h, w].
__global__ void upsample_kernel(const float *x, float *y, int height, int width, unsigned long long count) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int out_width = width * 2, out_height = height * 2;
    const int ox = (int)(i % out_width);
    const int oy = (int)((i / out_width) % out_height);
    const unsigned long long c = i / ((unsigned long long)out_width * out_height);
    y[i] = x[(c * height + oy / 2) * width + ox / 2];
}

extern "C" int qi_vae_upsample2x(const float *x, float *y, int channels, int height, int width) {
    const unsigned long long count = (unsigned long long)channels * height * width * 4;
    upsample_kernel<<<blocks_for(count, 256), 256>>>(x, y, height, width, count);
    return (int)cudaGetLastError();
}

// DupUp3D shortcut for the first chunk, added into y[out, 2H, 2W]: the input is
// repeated `repeats` times along channels and viewed as
// [out, factor_t, 2, 2]; the first chunk keeps temporal slice factor_t - 1.
// Output (oc, 2h + a, 2w + b) therefore reads input channel
// (oc * 4 * factor_t + (factor_t - 1) * 4 + a * 2 + b) / repeats.
__global__ void dup_up_add_kernel(float *y, const float *x, int height, int width, int factor_t, int repeats,
                                  unsigned long long count) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int out_width = width * 2, out_height = height * 2;
    const int ox = (int)(i % out_width);
    const int oy = (int)((i / out_width) % out_height);
    const int oc = (int)(i / ((unsigned long long)out_width * out_height));
    const int a = oy % 2, b = ox % 2;
    const int channel = (oc * 4 * factor_t + (factor_t - 1) * 4 + a * 2 + b) / repeats;
    y[i] += x[((size_t)channel * height + oy / 2) * width + ox / 2];
}

extern "C" int qi_vae_dup_up_add(float *y, const float *x, int out_channels, int height, int width, int factor_t,
                                 int repeats) {
    const unsigned long long count = (unsigned long long)out_channels * height * width * 4;
    dup_up_add_kernel<<<blocks_for(count, 256), 256>>>(y, x, height, width, factor_t, repeats, count);
    return (int)cudaGetLastError();
}

__global__ void add_kernel(float *y, const float *x, unsigned long long count) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < count) y[i] += x[i];
}

extern "C" int qi_vae_add(float *y, const float *x, unsigned long long count) {
    add_kernel<<<blocks_for(count, 256), 256>>>(y, x, count);
    return (int)cudaGetLastError();
}

__global__ void softmax_rows_kernel(float *scores, int columns, float scale) {
    __shared__ float scratch[32];
    float *row = scores + (size_t)blockIdx.x * columns;
    float peak = -INFINITY;
    for (int c = threadIdx.x; c < columns; c += blockDim.x) peak = fmaxf(peak, row[c] * scale);
    for (int o = 16; o > 0; o /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, o));
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = peak;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < blockDim.x / 32 ? scratch[threadIdx.x] : -INFINITY;
        for (int o = 16; o > 0; o /= 2) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) scratch[0] = v;
    }
    __syncthreads();
    peak = scratch[0];
    __syncthreads();
    float sum = 0.0f;
    for (int c = threadIdx.x; c < columns; c += blockDim.x) {
        const float e = expf(row[c] * scale - peak);
        row[c] = e;
        sum += e;
    }
    for (int o = 16; o > 0; o /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < blockDim.x / 32 ? scratch[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) scratch[0] = v;
    }
    __syncthreads();
    const float inverse = 1.0f / scratch[0];
    for (int c = threadIdx.x; c < columns; c += blockDim.x) row[c] *= inverse;
}

// Single-head attention over pixels for the mid block. qkv: [3C, N] (the
// to_qkv convolution output), out: [C, N]. scores: N * N floats.
// out[:, i] = sum_j softmax_j(q[:, i] . k[:, j] / sqrt(C)) v[:, j]
extern "C" int qi_vae_attention(const float *qkv, float *out, float *scores, int channels, int pixels) {
    if (vae_blas == nullptr && cublasCreate(&vae_blas) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorInitializationError;
    const float *q = qkv, *k = qkv + (size_t)channels * pixels, *v = qkv + 2 * (size_t)channels * pixels;
    const float one = 1.0f, zero = 0.0f;
    // Row-major S[i][j] = sum_c q[c][i] k[c][j]; column-major that is
    // S^T (N x N) = K^T-ish: C_col[j][i] = sum_c k[c][j] q[c][i].
    if (cublasSgemm(vae_blas, CUBLAS_OP_N, CUBLAS_OP_T, pixels, pixels, channels, &one, k, pixels, q, pixels, &zero,
                    scores, pixels) != CUBLAS_STATUS_SUCCESS) {
        return (int)cudaErrorUnknown;
    }
    softmax_rows_kernel<<<pixels, 256>>>(scores, pixels, 1.0f / sqrtf((float)channels));
    if (cudaError_t error = cudaGetLastError()) return (int)error;
    // Row-major out[c][i] = sum_j v[c][j] P[i][j]; column-major out^T (N x C):
    // C_col[i][c] = sum_j P_col^T... computed as P (row-major N x N) times V^T.
    if (cublasSgemm(vae_blas, CUBLAS_OP_T, CUBLAS_OP_N, pixels, channels, pixels, &one, scores, pixels, v, pixels,
                    &zero, out, pixels) != CUBLAS_STATUS_SUCCESS) {
        return (int)cudaErrorUnknown;
    }
    return (int)cudaGetLastError();
}

// Packed latents [N, 64] (row = pixel) to NCHW [64, N], un-normalized:
// z[c][p] = latents[p][c] * std[c] + mean[c].
__global__ void unpack_latents_kernel(const float *latents, const float *mean, const float *std, float *z, int pixels) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixels * 64) return;
    const int c = i / pixels, p = i % pixels;
    z[i] = latents[p * 64 + c] * std[c] + mean[c];
}

extern "C" int qi_vae_unpack_latents(const float *latents, const float *mean, const float *std, float *z, int pixels) {
    unpack_latents_kernel<<<blocks_for((unsigned long long)pixels * 64, 256), 256>>>(latents, mean, std, z, pixels);
    return (int)cudaGetLastError();
}

// Decoder output [4, H, W] to interleaved RGBA8: clamp to [-1, 1], then
// round-half-even((x * 0.5 + 0.5) * 255) as the pipeline's postprocess does.
__global__ void to_rgba_kernel(const float *x, unsigned char *rgba, int pixels) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    for (int c = 0; c < 4; ++c) {
        const float v = fminf(fmaxf(x[(size_t)c * pixels + p], -1.0f), 1.0f);
        const float unit = fminf(fmaxf(v * 0.5f + 0.5f, 0.0f), 1.0f);
        rgba[(size_t)p * 4 + c] = (unsigned char)rintf(unit * 255.0f);
    }
}

extern "C" int qi_vae_to_rgba(const float *x, unsigned char *rgba, int pixels) {
    to_rgba_kernel<<<blocks_for(pixels, 256), 256>>>(x, rgba, pixels);
    return (int)cudaGetLastError();
}
