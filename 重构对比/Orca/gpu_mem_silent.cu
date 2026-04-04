/* GPU/CPU 内存申请、拷贝和内存池初始化的实现，供整个最小运行链复用。 */
#include <chrono>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "gpu/helper_cuda.h"
#include "gpu/gpu_stats.h"
#include <cassert>

cudaMemPool_t mempool;

static uint64_t choosePoolReservationBytes()
{
    if (const char *envBytes = std::getenv("GPU_MEM_POOL_BYTES"))
    {
        const unsigned long long parsed = std::strtoull(envBytes, nullptr, 10);
        if (parsed > 0)
            return static_cast<uint64_t>(parsed);
    }

    cudaDeviceProp prop{};
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

    const uint64_t fourGiB = 4ULL << 30;
    const uint64_t halfTotal = static_cast<uint64_t>(prop.totalGlobalMem) / 2;
    return std::min(fourGiB, halfTotal);
}

extern "C" void initGPUMemPool()
{
    int isMemPoolSupported = 0;
    int device = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&isMemPoolSupported,
                                           cudaDevAttrMemoryPoolsSupported, device));
    assert(isMemPoolSupported);

    checkCudaErrors(cudaDeviceGetDefaultMemPool(&mempool, device));
    uint64_t threshold = UINT64_MAX;
    checkCudaErrors(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
    uint64_t *d_dummy_ptr;
    uint64_t bytes = choosePoolReservationBytes();
    checkCudaErrors(cudaMallocAsync(&d_dummy_ptr, bytes, 0));
    checkCudaErrors(cudaFreeAsync(d_dummy_ptr, 0));
}

extern "C" uint8_t *gpuMalloc(size_t size_in_bytes)
{
    uint8_t *d_a;
    checkCudaErrors(cudaMallocAsync(&d_a, size_in_bytes, 0));
    return d_a;
}

extern "C" uint8_t *cpuMalloc(size_t size_in_bytes, bool pin)
{
    uint8_t *h_a;
    int err = posix_memalign((void **)&h_a, 32, size_in_bytes);
    assert(err == 0 && "posix memalign");
    if (pin)
        checkCudaErrors(cudaHostRegister(h_a, size_in_bytes, cudaHostRegisterDefault));
    return h_a;
}

extern "C" void gpuFree(void *d_a)
{
    checkCudaErrors(cudaFreeAsync(d_a, 0));
}

extern "C" void cpuFree(void *h_a, bool pinned)
{
    if (pinned)
        checkCudaErrors(cudaHostUnregister(h_a));
    free(h_a);
}

extern "C" uint8_t *moveToCPU(uint8_t *d_a, size_t size_in_bytes, Stats *s)
{
    uint8_t *h_a = cpuMalloc(size_in_bytes, true);
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveIntoGPUMem(uint8_t *d_a, uint8_t *h_a, size_t size_in_bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveIntoCPUMem(uint8_t *h_a, uint8_t *d_a, size_t size_in_bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveToGPU(uint8_t *h_a, size_t size_in_bytes, Stats *s)
{
    uint8_t *d_a = gpuMalloc(size_in_bytes);
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return d_a;
}
