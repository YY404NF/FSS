#pragma once

#include "gpu/gpu_data_types.h"
#include "gpu/gpu_random.h"
#include "gpu/packing_utils.h"
#include "gpu_dpf_templates.h"

typedef void (*genShares)(u64 x, u8 *tab);

struct GPUSSTabKey
{
    // 小 bin 场景下直接使用 SSTable；ss 指向每个输入各自的一张压缩查找表。
    int bin, N;
    u8 *ss;
    u64 memSzSS, memSzOut;
};

GPUSSTabKey readGPUSSTabKey(u8 **key_as_bytes)
{
    // 从序列化 key 中切出 SSTable 视图，不额外复制表内容。
    GPUSSTabKey k;
    memcpy(&k, *key_as_bytes, 2 * sizeof(int));
    *key_as_bytes += 2 * sizeof(int);
    k.ss = *key_as_bytes;
    k.memSzSS = k.N * (1ULL << (max(0, k.bin - 3))); // size in bytes
    *key_as_bytes += k.memSzSS;
    k.memSzOut = ((k.N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    return k;
}

__device__ u8 lookup(u8 *ss, u64 x)
{
    return u8((ss[x / 8] >> (x % 8)) & u8(1));
}

template <typename T, int E, dpfPrologue pr, dpfEpilogue ep>
__global__ void lookupSSTable(int party, int bin, int N,
                              T *in, u8 *ss, u32 *out)
{
    // 每个线程处理一个输入，先按 prologue 生成查询点，再在本地表中逐项查值。
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        auto x = u64(in[tid]);
        gpuMod(x, bin);
        int tabSz = (1ULL << (max(0, bin - 3))); // number of bytes occupied by the table
        // printf("Table start idx %d=%d\n", tid, tabSz * tid);
        u8 *localSS = &ss[tid * tabSz];
        u64 x1[E];
        // populate the input
        pr(party, bin, N, x, x1);
        u8 o[E];
        for (int e = 0; e < E; e++)
        {
            gpuMod(x1[e], bin);
            // printf("X[%d]=%ld, %ld, %d, %d\n", e, x, x1[e], int(lookup(localSS, x1[e])), int(localSS[x1[e] / 8]));
            o[e] = lookup(localSS, x1[e]);
        }
        ep(party, bin, N, x, o, out, 0);
    }
}


template <typename T, int E, dpfPrologue pr, dpfEpilogue ep>
u32 *gpuLookupSSTable(GPUSSTabKey &k, int party, T *d_in, Stats* s, std::vector<u32 *> *h_masks=NULL)
{
    // 将 CPU 侧保存的表搬到 GPU，执行查表，再返回打包后的输出缓冲区。
    auto d_out = moveMasks(k.memSzOut, h_masks, s);
    // printf("Bin=%d, Memsz=%ld\n", k.bin, k.memSzSS);
    auto d_ss = (u8 *)moveToGPU((u8 *)k.ss, k.memSzSS, s);
    lookupSSTable<T, E, pr, ep><<<(k.N - 1) / 128 + 1, 128>>>(party, k.bin, k.N, d_in, d_ss, d_out);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_ss);
    return d_out;
}

__device__ void dpfShares(u64 x, u8 *tab)
{
    // DPF 小 bin 场景只需要在目标位置翻转一个比特。
    int idx = x / 8;
    u8 c = u8(1) << (x % 8);
    tab[idx] ^= c;
}

__device__ void dcfShares(u64 x, u8 *tab)
{
    // DCF 小 bin 场景需要把阈值之前的前缀区间编码到表里。
    int idx = x / 8;
    // printf("Rin=%ld, tab[0]=%d, %d\n", x, tab[0], idx);
    for (int i = 0; i < idx; i++)
    {
        tab[i] ^= u8(-1);
    }
    int off = x % 8;
    if (off)
    {
        u8 c = u8((1 << off) - 1);
        tab[idx] ^= c;
    }
    // printf("Rin=%ld, tab[0]=%d\n", x, tab[0]);
}

template <typename T, genShares g>
__global__ void genSSTableKernel(int party, int bin, int N, T *rin, u8 *tab)
{
    // 每个线程根据自己的 rin 生成一张局部 SSTable。
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        T x = rin[tid];
        // printf("SSTab %d: %ld\n", tid, x);
        gpuMod(x, bin);
        // printf("SSTab %d: %ld\n", tid, x);
        u64 tabSz = (1ULL << (max(0, bin - 3)));
        g(u64(x), &tab[tabSz * tid]);
    }
}

template <typename T, genShares g>
void genSSTable(uint8_t **key_as_bytes, int party, int bin, int N, T *d_rin)
{
    // 生成 SSTable 并直接写入序列化 key 缓冲区，供小 bin 路径复用。
    writeInt(key_as_bytes, bin);
    writeInt(key_as_bytes, N);
    // printf("%d, %d\n", bin, N);
    u64 memSize = N * (1ULL << (max(0, bin - 3)));
    auto d_share0 = randomGEOnGpu<u8>(memSize, 8);
    // printf("here!!!!!!!!!!!!!!\n");
    if (party == SERVER1)
        genSSTableKernel<T, g><<<(N - 1) / 128 + 1, 128>>>(party, bin, N, d_rin, d_share0);
    moveIntoCPUMem(*key_as_bytes, d_share0, memSize, NULL);
    *key_as_bytes += memSize;
    gpuFree(d_share0);
}
