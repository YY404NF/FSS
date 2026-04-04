/* 打包写回与 key 字节流序列化工具，供 DPF/DCF 主链共享。 */
#pragma once

#include <cstring>

#include "gpu/gpu_data_types.h"

inline void writeInt(u8 **key_as_bytes, int value)
{
    std::memcpy(*key_as_bytes, &value, sizeof(int));
    *key_as_bytes += sizeof(int);
}

template <typename T>
__device__ void writePackedOp(u32 *d_output, T value, int bout, u64 n)
{
    u64 threadId = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    unsigned mask = __ballot_sync(FULL_MASK, threadId < n);
    int laneId = threadIdx.x & 0x1f;
    if (bout == 1)
    {
        int packed = static_cast<int>(value);
        packed <<= laneId;
        for (int j = 16; j >= 1; j /= 2)
            packed += __shfl_down_sync(mask, packed, j, 32);
        if (laneId == 0)
            d_output[threadId / 32] = static_cast<u32>(packed);
    }
    else if (bout == 2)
    {
        auto packed = u64(value);
        packed <<= (2 * laneId);
        for (int j = 16; j >= 1; j /= 2)
            packed += __shfl_down_sync(mask, packed, j, 32);
        if (laneId == 0)
        {
            d_output[threadId / 16] = static_cast<u32>(packed);
            if (n - threadId > 16)
                d_output[threadId / 16 + 1] = static_cast<u32>(packed >> 32);
        }
    }
    else
    {
        ((T *)d_output)[threadId] = value;
    }
}
