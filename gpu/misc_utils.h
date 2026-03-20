/* 杂项 GPU/FSS 工具集合，当前主要保留位宽裁剪、线性组合和 share 打包辅助。 */
#pragma once

#include <cassert>
#include <cstdint>

#include "gpu/gpu_data_types.h"
#include "gpu/helper_cuda.h"
#include "gpu/packing_utils.h"

template <typename T>
__device__ inline void gpuMod(T &x, int bw)
{
    if (bw < sizeof(T) * 8)
        x &= ((T(1) << bw) - 1);
}

template <typename T>
__device__ void linearComb(int i, T c, T d_A)
{
    static_assert(std::is_arithmetic<T>::value, "Only arithmetic types supported");
    return c * d_A[i];
}

template <typename T, typename... Args>
__device__ T linearComb(int i, T c)
{
    // static_assert(std::is_arithmetic<T>::value, "Only arithmetic types supported");
    return c;
}

template <typename T, typename... Args>
__device__ T linearComb(int i, T c, T *A)
{
    // static_assert(std::is_arithmetic<T>::value, "Only arithmetic types supported");
    // if(i == 0) printf("Linear comb: %ld, %ld\n", c, A[i]);
    return c * A[i];
}

template <typename T, typename... Args>
__device__ T linearComb(int i, T c, T *A, Args... args)
{
    // if(i == 0) printf("Linear comb: %ld, %ld\n", c, A[i]);
    // static_assert(std::is_arithmetic<T>::value, "Only arithmetic types supported");
    return c * A[i] + linearComb(i, args...);
}

template <typename T, typename... Args>
__global__ void linearCombWrapper(int bw, int N, T *d_O, Args... args)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        d_O[i] = linearComb(i, args...);
        gpuMod(d_O[i], bw);
        // if(i == 0) printf("Op=%lu\n", d_O[i]);
    }
}

template <typename T, typename... Arguments>
void gpuLinearComb(int bw, int N, T *d_O, Arguments... args)
{
    const int thread_block_size = 128;
    linearCombWrapper<<<(N - 1) / thread_block_size + 1, thread_block_size>>>(bw, N, d_O, args...);
    checkCudaErrors(cudaDeviceSynchronize());
    // checkCudaErrors(cudaGetLastError());
}

template <typename T>
__global__ void modKernel(u64 N, T *d_data, int bw)
{
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i < N)
    {
        gpuMod(d_data[i], bw);
    }
}

template <typename TIn, typename TOut>
__global__ void getPackedSharesKernel(u64 N, int party, TIn *d_A, TOut *d_A0, u32 *d_packed_A, int bw)
{
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i < N)
    {
        TOut share_A = TOut(d_A[i]);
        if (d_A0)
            share_A = party == SERVER0 ? d_A0[i] : TOut(d_A[i]) - d_A0[i];
        gpuMod(share_A, bw);
        // if(i == 1) printf("%lu: share_A = %u, %u, %d\n", i, share_A, d_A[i], party);
        writePackedOp(d_packed_A, share_A, bw, N);
    }
}
