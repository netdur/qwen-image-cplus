// C ABI over the CUDA runtime for the C+ engine. Every function returns a
// cudaError_t value (0 is success) so the C+ side can report failures.

#include <cuda_runtime.h>

extern "C" int qi_cuda_device(char *name, int name_capacity, int *major, int *minor,
                              unsigned long long *memory_bytes, int *multiprocessors) {
    cudaDeviceProp prop;
    cudaError_t status = cudaGetDeviceProperties(&prop, 0);
    if (status != cudaSuccess) return (int)status;
    int i = 0;
    for (; i + 1 < name_capacity && prop.name[i] != '\0'; ++i) name[i] = prop.name[i];
    if (name_capacity > 0) name[i] = '\0';
    *major = prop.major;
    *minor = prop.minor;
    *memory_bytes = (unsigned long long)prop.totalGlobalMem;
    *multiprocessors = prop.multiProcessorCount;
    return 0;
}

extern "C" int qi_cuda_malloc(void **pointer, unsigned long long bytes) {
    return (int)cudaMalloc(pointer, (size_t)bytes);
}

extern "C" int qi_cuda_free(void *pointer) {
    return (int)cudaFree(pointer);
}

extern "C" int qi_cuda_upload(void *device, const void *host, unsigned long long bytes) {
    return (int)cudaMemcpy(device, host, (size_t)bytes, cudaMemcpyHostToDevice);
}

extern "C" int qi_cuda_download(void *host, const void *device, unsigned long long bytes) {
    return (int)cudaMemcpy(host, device, (size_t)bytes, cudaMemcpyDeviceToHost);
}

extern "C" int qi_cuda_synchronize(void) {
    return (int)cudaDeviceSynchronize();
}

extern "C" const char *qi_cuda_error_name(int status) {
    return cudaGetErrorName((cudaError_t)status);
}

__global__ static void axpy_kernel(float *y, const float *x, float a, unsigned long long count) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < count) y[i] = a * x[i] + y[i];
}

// Smoke kernel for the build and link path: y = a * x + y on the device.
extern "C" int qi_cuda_axpy(float *y, const float *x, float a, unsigned long long count) {
    axpy_kernel<<<(unsigned)((count + 255) / 256), 256>>>(y, x, a, count);
    return (int)cudaGetLastError();
}

extern "C" int qi_cuda_memory(unsigned long long *free_bytes, unsigned long long *total_bytes) {
    size_t free_value = 0, total_value = 0;
    cudaError_t status = cudaMemGetInfo(&free_value, &total_value);
    *free_bytes = free_value;
    *total_bytes = total_value;
    return (int)status;
}

// Streams and events for overlapping weight uploads with compute. A null
// stream means the legacy default stream, where every kernel here runs.
extern "C" int qi_cuda_stream_create(void **stream) {
    return (int)cudaStreamCreateWithFlags((cudaStream_t *)stream, cudaStreamNonBlocking);
}

extern "C" int qi_cuda_stream_destroy(void *stream) {
    return (int)cudaStreamDestroy((cudaStream_t)stream);
}

extern "C" int qi_cuda_event_create(void **event) {
    return (int)cudaEventCreateWithFlags((cudaEvent_t *)event, cudaEventDisableTiming);
}

extern "C" int qi_cuda_event_destroy(void *event) {
    return (int)cudaEventDestroy((cudaEvent_t)event);
}

extern "C" int qi_cuda_event_record(void *event, void *stream) {
    return (int)cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
}

extern "C" int qi_cuda_stream_wait(void *stream, void *event) {
    return (int)cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, 0);
}

extern "C" int qi_cuda_upload_async(void *device, const void *host, unsigned long long bytes, void *stream) {
    return (int)cudaMemcpyAsync(device, host, (size_t)bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);
}

// Blocks the host until `event` has completed.
extern "C" int qi_cuda_event_synchronize(void *event) {
    return (int)cudaEventSynchronize((cudaEvent_t)event);
}

// Page-locks existing host memory (cheap once its pages are resident), and
// releases it again.
extern "C" int qi_cuda_host_register(void *pointer, unsigned long long bytes) {
    return (int)cudaHostRegister(pointer, (size_t)bytes, cudaHostRegisterDefault);
}

extern "C" int qi_cuda_host_unregister(void *pointer) {
    return (int)cudaHostUnregister(pointer);
}
