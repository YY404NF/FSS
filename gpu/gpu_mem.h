#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
// #include <sys/types.h>
// extern cudaMemPool_t mempool;

class Stats;

// 这一层统一封装最小运行链用到的主机/GPU 内存申请与拷贝接口。
extern "C" uint8_t *gpuMalloc(size_t size_in_bytes);
extern "C" uint8_t *cpuMalloc(size_t size_in_bytes, bool pin = true);
extern "C" void cpuFree(void *h_a, bool pinned = true);
extern "C" void gpuFree(void *d_a);
extern "C" uint8_t *moveToGPU(uint8_t *h_a, size_t size_in_bytes, Stats *);
extern "C" uint8_t *moveIntoGPUMem(uint8_t *d_a, uint8_t *h_a, size_t size_in_bytes, Stats *s);
extern "C" uint8_t *moveToCPU(uint8_t *d_a, size_t size_in_bytes, Stats *);
extern "C" uint8_t *moveIntoCPUMem(uint8_t *h_a, uint8_t *d_a, size_t size_in_bytes, Stats *s);
// 初始化当前工程使用的 GPU 内存池，供后续申请复用。
extern "C" void initGPUMemPool();
