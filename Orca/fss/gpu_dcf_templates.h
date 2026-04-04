#pragma once

#include "gpu/gpu_data_types.h"
#include "gpu/misc_utils.h"
#include "gpu/packing_utils.h"

#include "fss/gpu_fss_helper.h"

// using namespace std;
namespace dcf
{
    // DCF 通过 prologue/epilogue 模板把同一条比较主链复用到不同语义上。
    typedef void (*dcfPrologue)(int party, int bin, int N,
                                u64 x,
                                u64 *o);
    typedef void (*dcfEpilogue)(int party, int bin, int bout, int N,
                                u64 x,
                                u64 *o_l, u32 *out_g, u64 oStride);

    // 默认前处理：直接比较原始输入。
    __device__ void idPrologue(int party, int bin, int N,
                               u64 x,
                               u64 *o)
    {
        o[0] = x;
    }

    // 默认后处理：直接把 DCF 输出按 bout 位宽写入打包缓冲区。
    __device__ void idEpilogue(int party, int bin, int bout, int N,
                               u64 x,
                               u64 *o_l, u32 *out_g, u64 oStride)
    {
        auto o1 = u64(*o_l);
        writePackedOp(out_g, o1, bout, N);
    }

    // mask 后处理：把新输出和已有 mask 相加后再按 bout 截断。
    __device__ void maskEpilogue(int party, int bin, int bout, int N,
                                 u64 x,
                                 u64 *o_l, u32 *out_g, u64 oStride)
    {
        auto o1 = u64(*o_l);
        auto mask = getVCW(bout, out_g, N, 0);
        // printf("Mask: %ld, output: %ld\n", mask, o);
        o1 = o1 + mask;
        gpuMod(o1, bout);
        writePackedOp(out_g, o1, bout, N);
    }

    // dReLU 前处理会同时构造原值和加半区间偏移后的值，供后面合成符号结果。
    __device__ void dReluPrologue(int party, int bin, int N,
                                  u64 x,
                                  u64 *o)
    {
        o[0] = x;
        o[1] = (x + (1ULL << (bin - 1)));
    }

    template <bool returnXLtRin>
    // dReLU 后处理负责把两路比较结果折叠成最终输出，并可选返回 x<rin 的辅助结果。
    __device__ void dReluEpilogue(int party, int bin, int bout, int N,
                                  u64 x,
                                  u64 *o_l, u32 *out_g, u64 oStride)
    {
        auto o1 = o_l[0];
        auto o2 = o_l[1];
        auto mask = getVCW(bout, out_g, N, 0);
        auto o = o2 - o1 + mask;
        // printf("o1=%lu, o2=%lu, mask=%lu, o=%lu\n", o1, o2, mask, o);
        if (party == SERVER1)
        {
            auto x2 = (x + (1ULL << (bin - 1)));
            gpuMod(x2, bin);
            o += (x2 >= (1ULL << (bin - 1)));
        }
        gpuMod(o, bout);
        writePackedOp(out_g, o, bout, N);
        // writeVCW(bout, out_g, o, 0, N);
        if (returnXLtRin)
        {
            o1 += getVCW(bout, out_g + oStride, N, 0);
            gpuMod(o1, bout);
            writePackedOp(out_g + oStride, o1, bout, N);
            // writeVCW(bout, out_g, o1, 1, N);
        }
    }

}
