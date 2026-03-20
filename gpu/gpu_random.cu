/*
 * This program uses the host CURAND API to generate 100
 * pseudorandom floats.
 */
#include <cstdint>

#include <cuda_runtime.h>

#include "gpu/curand_utils.h"
#include "gpu/gpu_data_types.h"
#include "gpu/gpu_mem.h"

#include "gpu/gpu_random.h"

// using data_type = u32;
// cudaStream_t stream = NULL;
curandGenerator_t gpuGen[2];
curandOrdering_t order = CURAND_ORDERING_PSEUDO_BEST;

template <typename T>
__global__ void randomModKernel(u64 n, T *d_data, int bw)
{
  u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (i < n && bw < sizeof(T) * 8)
  {
    d_data[i] &= ((T(1) << bw) - 1);
  }
}

void randomUIntsOnGpu(const u64 n, u32 *d_data)
{
  // 直接向 GPU 缓冲区写入原始 32-bit 随机数。
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  CURAND_CHECK(curandGenerate(gpuGen[device], d_data, n));
}

template <typename T>
T *randomGEOnGpu(const u64 n, int bw)
{
  // 先生成足量原始随机数，再通过本地 kernel 截断到目标位宽。
  u64 numUInts = (n * sizeof(T) - 1) / (sizeof(u32)) + 1;
  // printf("random n=%lu, ints=%lu, bw=%d\n", n, numUInts, bw);
  // assert((n * sizeof(T)) % sizeof(u32) == 0);
  auto d_data = (u32 *)gpuMalloc(numUInts * sizeof(u32));
  randomUIntsOnGpu(/*2 * n*/ numUInts, /*(u32*)*/ d_data);
  randomModKernel<<<(n - 1) / 256 + 1, 256>>>(n, (T *)d_data, bw);
  return (T *)d_data;
}

AESBlock *randomAESBlockOnGpu(const int n)
{
  // AESBlock 本质上也是随机比特串，这里直接按 4 个 u32 填满每个 block。
  AESBlock *d_data = (AESBlock *)gpuMalloc(n * sizeof(AESBlock));
  randomUIntsOnGpu(4 * n, (u32 *)d_data);
  return d_data;
}

void initGPURandomness()
{
  // 当前最小集合固定使用 XORWOW + 固定 seed，便于稳定复现。
  const unsigned long long offset = 0ULL;
  const unsigned long long seed = 12345ULL;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  CURAND_CHECK(curandCreateGenerator(&(gpuGen[device]), CURAND_RNG_PSEUDO_XORWOW));
  CURAND_CHECK(curandSetGeneratorOffset(gpuGen[device], offset));
  CURAND_CHECK(curandSetGeneratorOrdering(gpuGen[device], order));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gpuGen[device], seed));
}

void destroyGPURandomness()
{
  // 与当前 device 绑定的 CURAND generator 在 Runtime 析构阶段统一释放。
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  CURAND_CHECK(curandDestroyGenerator(gpuGen[device]));
}
