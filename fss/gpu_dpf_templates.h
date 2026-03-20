#pragma once

#include "gpu/misc_utils.h"
#include "gpu/packing_utils.h"
#include "gpu/gpu_data_types.h"

#include "gpu_fss_helper.h"

typedef void (*dpfPrologue)(int party, int bin, int N,
                            u64 x,
                            u64 *o);
typedef void (*dpfEpilogue)(int party, int bin, int N,
                            u64 x,
                            u8 *o, u32 *out, u64 oStride);

// 默认前处理：直接把输入传给后续 DPF 树求值。
__device__ void idPrologue(int party, int bin, int N,
                           u64 x,
                           u64 *o)
{
    // printf("Inside truncate=%ld\n", x);
    o[0] = x;
    // gpuMod(o[0], bin);
}

template <u64 p>
// dReLU 变体会先把输入平移到以阈值 p 为中心的比较形式。
__device__ void dReluPrologue(int party, int bin, int N,
                              u64 x,
                              u64 *o)
{
    o[0] = p - x - 1;
}

template <u64 p, u64 q>
// GELU 相关路径会一次生成多个比较点，后续在 epilogue 中合成结果。
__device__ void geluPrologue(int party, int bin, int N,
                             u64 x,
                             u64 *o)
{
    o[0] = -x - 1;
    o[1] = p - x - 1;
    o[2] = q - x - 1;
}

template <u64 p, bool flip>
// dReLU 后处理：把 DPF 输出与已有 mask 以及 SERVER1 的本地修正合并。
__device__ void dReluEpilogue(int party, int bin, int N,
                              u64 x,
                              u8 *o, u32 *out, u64 oStride)
{
    auto o1 = u64(*o);
    auto mask = getVCW(1, out, N, 0);
    o1 ^= mask;
    if (party == SERVER1)
        o1 ^= (gpuMsb(x - p, bin + 1) ^ u64(flip));
    // gpuMod(o, 1);
    //  ^ o ^ mask;
    // printf("Epilogue: %ld, %ld, %ld\n", mask, u64(*o), gpuMsb(x, bin + 1));
    writePackedOp(out, o1, 1, N);
}

template <u64 p, u64 q>
// GELU 后处理会把多路比较结果重新折叠成 dReLU / 区间比较两路输出。
__device__ void geluEpilogue(int party, int bin, int N,
                             u64 x,
                             u8 *o, u32 *out, u64 oStride)
{
    auto o1 = u64(o[0]);
    auto dReluMask = getVCW(1, out, N, 0);
    o1 ^= dReluMask;
    if (party == SERVER1)
        o1 ^= gpuMsb(x, bin + 1);
    // gpuMod(o, 1);
    //  ^ o ^ mask;
    // printf("Epilogue: %lu, %lu\n", dReluMask, o1);
    writePackedOp(out, o1, 1, N);
    // writeVCW(1, out, o1, 0, oStride);

    o1 = u64(o[1]);
    auto o2 = u64(o[2]);
    auto icMask = getVCW(1, out + oStride, N, 0);
    o1 ^= (o2 ^ icMask);
    if (party == SERVER1)
    {
        auto xp = x - p;
        // gpuMod(xp, bin + 1);
        auto xq = x - q;
        // gpuMod(xq, bin + 1);
        o1 ^= (gpuMsb(xp, bin + 1) ^ gpuMsb(xq, bin + 1));
    }
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if(i == 0) printf("Epilogue %d: %ld, %ld\n", i, mask, o, gpuMsb(x, bin + 1));
    // gpuMod(o, 1);
    //  ^ o ^ mask;
    //
    // writeVCW(1, out, o1, 1, N);
    writePackedOp(out + oStride, o1, 1, N);
    // printf("icMask=%lu, ic=%lu\n", icMask, o1);

}

// 仅把新输出叠加到已有 mask 上，不做额外语义变换。
__device__ void maskEpilogue(int party, int bin, int N,
                             u64 x,
                             u8 *o, u32 *out, u64 oStride)
{
    auto o1 = u64(*o);
    auto mask = getVCW(1, out, N, 0);
    // printf("Mask: %ld, output: %ld\n", mask, o);
    o1 = o1 ^ mask;
    writePackedOp(out, o1, 1, N);
}

// 默认后处理：直接把单比特结果写入打包输出。
__device__ void idEpilogue(int party, int bin, int N,
                           u64 x,
                           u8 *o, u32 *out, u64 oStride)
{

    writePackedOp(out, u64(*o), 1, N);
}
