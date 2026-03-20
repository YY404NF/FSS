#pragma once

#include "gpu/gpu_data_types.h"

// 生成 n 个指定位宽的随机元素，结果保存在 GPU 侧。
template <typename T>
T *randomGEOnGpu(const u64 n, int bw);
// 生成 n 个随机 AES block，供 DPF/DCF keygen 过程使用。
AESBlock *randomAESBlockOnGpu(const int n);
// 初始化和销毁最小运行链共享的 GPU 随机源。
void initGPURandomness();
void destroyGPURandomness();


#include "gpu_random.cu"
