#pragma once

#include "gpu/gpu_data_types.h"
#include "aes/gpu_aes_shm.h"
#include "gpu_sstab.h"

// using u32 = u32;

struct GPUDPFTreeKey
{
    // 单个批次的 DPF tree key 视图，字段直接指向序列化 key blob 内部的各段内存。
    int bin, N, evalAll;
    AESBlock *scw;
    AESBlock *l0, *l1;
    u32 *tR;
    u64 szScw, memSzScw, memSzL, memSzT, memSzOut;
};

struct GPUDPFKey
{
    // 小 bin 走 SSTable，大 bin 走 tree key；这里统一描述两种布局的入口信息。
    int bin, M, B;
    u64 memSzOut;
    GPUDPFTreeKey *dpfTreeKey;
    GPUSSTabKey ssKey;
};

GPUDPFTreeKey readGPUDPFTreeKey(u8 **key_as_bytes)
{
    // 按当前序列化格式顺序切片，构造单个 tree key 的只读视图。
    GPUDPFTreeKey k;

    std::memcpy((char *)&k, *key_as_bytes, 3 * sizeof(int));
    *key_as_bytes += 3 * sizeof(int);

    k.szScw = k.N * (k.bin - LOG_AES_BLOCK_LEN);
    k.memSzScw = k.szScw * sizeof(AESBlock);
    k.scw = (AESBlock *)*key_as_bytes;

    *key_as_bytes += k.memSzScw;
    k.memSzL = k.N * sizeof(AESBlock);
    k.l0 = (AESBlock *)*key_as_bytes;
    *key_as_bytes += k.memSzL;
    k.l1 = (AESBlock *)*key_as_bytes;
    *key_as_bytes += k.memSzL;

    if (k.evalAll)
        k.memSzT = k.N * sizeof(u32);
    else
        k.memSzT = ((k.N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE) * (k.bin - LOG_AES_BLOCK_LEN);
    k.tR = (u32 *)*key_as_bytes;
    *key_as_bytes += k.memSzT;
    k.memSzOut = ((k.N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    return k;
}

GPUDPFKey readGPUDPFKey(u8 **key_as_bytes)
{
    // 解析整份 DPF key，根据 bin 自动选择 SSTable 或 tree key 布局。
    GPUDPFKey k;
    k.bin = *((int *)*key_as_bytes);
    if (k.bin <= 7)
    {
        k.ssKey = readGPUSSTabKey(key_as_bytes);
        k.M = k.ssKey.N;
        k.B = 1;
        k.memSzOut = k.ssKey.memSzOut;
    }
    else
    {
        memcpy(&k, *key_as_bytes, 3 * sizeof(int));
        *key_as_bytes += (3 * sizeof(int));

        k.dpfTreeKey = new GPUDPFTreeKey[k.B];
        k.memSzOut = 0;
        for (int b = 0; b < k.B; b++)
        {
            k.dpfTreeKey[b] = readGPUDPFTreeKey(key_as_bytes);
            k.memSzOut += k.dpfTreeKey[b].memSzOut;
        }
    }
    return k;
}

// 兼容旧调用点保留的别名；当前名字虽然容易混淆，但暂不改行为。
const auto readGPUDcfKey = readGPUDPFKey;

#include "gpu_dpf.cu"
