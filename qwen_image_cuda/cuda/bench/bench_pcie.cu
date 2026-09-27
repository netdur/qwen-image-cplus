// Host-to-device bandwidth from pinned memory, the rate at which FP16
// transformer blocks could stream into a small GPU each step.
//
//   nvcc -O3 -arch=sm_75 -o bench_pcie bench_pcie.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

int main() {
    const size_t block_bytes = 436224000;  // one all-FP16 transformer block
    const int copies = 16;
    void *host = nullptr, *device = nullptr;
    if (cudaMallocHost(&host, block_bytes) != cudaSuccess || cudaMalloc(&device, block_bytes) != cudaSuccess) {
        std::fprintf(stderr, "allocation failed\n");
        return 1;
    }
    std::memset(host, 1, block_bytes);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMemcpy(device, host, block_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < copies; ++i) cudaMemcpyAsync(device, host, block_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double gbps = (double)block_bytes * copies / (ms * 1e-3) / 1e9;
    std::printf("pinned H2D: %.2f GB/s, %.1f ms per 436 MB block, %.2f s per 32-block step\n",
                gbps, ms / copies, ms / copies * 32 / 1000.0);
    cudaFreeHost(host);
    cudaFree(device);
    return 0;
}
