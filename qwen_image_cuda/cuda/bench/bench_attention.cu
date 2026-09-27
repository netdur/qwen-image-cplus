// Compares the engine's attention (qi_attention_half: FP16 scores softmaxed in
// place, K/V converted once per block) with FP32 materialized scores
// (qi_attention_f16) at the real 1024x1024 step shape (4096 image queries,
// 32 + 4096 keys, 32 heads).
//
// A fused WMMA flash kernel (32-query x 32-key tiles, output accumulator in
// shared memory) was also measured here and removed: 131 ms per block at this
// shape against 61.6 ms materialized and 37.1 ms for qi_attention_half.
//
//   nvcc -O3 -arch=sm_75 -std=c++17 -o bench_attention bench_attention.cu ../blas.cu -lcublas

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" int qi_to_half(const float *source, void *out, unsigned long long count);
extern "C" int qi_attention_half(const float *q, const void *k_half, const void *v_half, float *out, void *scratch,
                                 int queries, int keys, int heads, int query_offset, int causal_rows,
                                 const int *visible_keys);
extern "C" int qi_attention_f16(const float *q, const float *k, const float *v, float *out, float *scores, void *halves,
                                int queries, int keys, int heads, int query_offset, int causal_rows);

static unsigned hash(unsigned x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }

int main(int argc, char **argv) {
    const int queries = argc > 1 ? std::atoi(argv[1]) : 4096, text = 32, keys = queries + text, heads = 32;
    const size_t width = (size_t)heads * 128;
    std::vector<float> host_q(queries * width), host_k(keys * width), host_v(keys * width);
    // RMS-normalized-like magnitudes (unit variance) as Q/K see after norm and RoPE.
    for (size_t i = 0; i < host_q.size(); ++i) host_q[i] = ((int)(hash((unsigned)i) % 2001) - 1000) * 0.0017f;
    for (size_t i = 0; i < host_k.size(); ++i) host_k[i] = ((int)(hash((unsigned)i * 7u + 1) % 2001) - 1000) * 0.0017f;
    for (size_t i = 0; i < host_v.size(); ++i) host_v[i] = ((int)(hash((unsigned)i * 13u + 5) % 2001) - 1000) * 0.001f;
    float *q, *k, *v, *reference, *scores; void *halves2;
    const int chunk = 1024;
    cudaMalloc(&q, host_q.size() * 4); cudaMalloc(&k, host_k.size() * 4); cudaMalloc(&v, host_v.size() * 4);
    cudaMalloc(&reference, host_q.size() * 4);
    cudaMalloc(&scores, (size_t)heads * chunk * keys * 4);
    cudaMalloc(&halves2, ((chunk + 2 * (size_t)keys) * width + (size_t)heads * chunk * keys) * 2);
    cudaMemcpy(q, host_q.data(), host_q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(k, host_k.data(), host_k.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(v, host_v.data(), host_v.size() * 4, cudaMemcpyHostToDevice);

    auto run_reference = [&]() {
        for (int c = 0; c < queries; c += chunk)
            qi_attention_f16(q + c * width, k, v, reference + c * width, scores, halves2, std::min(chunk, queries - c), keys,
                             heads, text + c, text);
    };
    float *half_out; void *kv_half, *half_scratch;
    cudaMalloc(&half_out, host_q.size() * 4);
    cudaMalloc(&kv_half, 2 * (size_t)keys * width * 2);
    cudaMalloc(&half_scratch, ((size_t)chunk * width + (size_t)heads * chunk * keys) * 2);
    auto run_half = [&]() {
        qi_to_half(k, kv_half, (unsigned long long)keys * width);
        qi_to_half(v, (char *)kv_half + (size_t)keys * width * 2, (unsigned long long)keys * width);
        for (int c = 0; c < queries; c += chunk)
            qi_attention_half(q + c * width, kv_half, (char *)kv_half + (size_t)keys * width * 2, half_out + c * width,
                              half_scratch, std::min(chunk, queries - c), keys, heads, text + c, text, nullptr);
    };
    run_reference(); run_half();
    cudaDeviceSynchronize();
    std::vector<float> b(host_q.size());
    cudaMemcpy(b.data(), reference, b.size() * 4, cudaMemcpyDeviceToHost);
    double energy = 0;
    for (size_t i = 0; i < b.size(); ++i) energy += (double)b[i] * b[i];
    auto time = [&](auto run) {
        run();
        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start); for (int i = 0; i < 5; ++i) run(); cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms = 0; cudaEventElapsedTime(&ms, start, stop); return ms / 5;
    };
    std::vector<float> h(host_q.size());
    cudaMemcpy(h.data(), half_out, h.size() * 4, cudaMemcpyDeviceToHost);
    double half_error = 0;
    for (size_t i = 0; i < h.size(); ++i) half_error += (h[i] - b[i]) * (double)(h[i] - b[i]);
    const float half_ms = time(run_half);
    std::printf("fp16 scores: %.2f ms per block, x32 %.2f s, nRMSE vs materialized FP32 scores %.2e\n", half_ms,
                half_ms * 32 / 1000, std::sqrt(half_error / energy));
    const float reference_ms = time(run_reference);
    std::printf("queries %d keys %d: materialized FP32 scores %.2f ms per block, x32 %.2f s\n", queries, keys,
                reference_ms, reference_ms * 32 / 1000);
    return 0;
}
