#pragma once

#include <stdexcept>
#include <vector>

#include "runtime/standalone_runtime.h"
#include "fss/gpu_dpf.h"

class Stats;

namespace gpu_mpc
{
namespace standalone
{
    // 生成 DPF 两方密钥，并将结果封装为可独立保存/传递的二进制 blob。
    template <typename T>
    std::pair<KeyBlob, KeyBlob> generateDpfKeys(Runtime &runtime, int bin, const std::vector<T> &rin, bool evalAll = false)
    {
        if (rin.empty())
            throw std::invalid_argument("rin must not be empty.");

        const std::size_t keyBytes = detail::dpfKeyBytes(bin, static_cast<int>(rin.size()), evalAll);
        KeyBlob key0(keyBytes);
        KeyBlob key1(keyBytes);

        T *d_rin = detail::copyVectorToGpu(rin);

        resetGPURandomness();
        u8 *cursor0 = key0.data();
        gpuKeyGenDPF(&cursor0, SERVER0, bin, static_cast<int>(rin.size()), d_rin, runtime.aes(), evalAll);
        key0.setSize(static_cast<std::size_t>(cursor0 - key0.data()));

        resetGPURandomness();
        u8 *cursor1 = key1.data();
        gpuKeyGenDPF(&cursor1, SERVER1, bin, static_cast<int>(rin.size()), d_rin, runtime.aes(), evalAll);
        key1.setSize(static_cast<std::size_t>(cursor1 - key1.data()));

        gpuFree(d_rin);
        return {std::move(key0), std::move(key1)};
    }

    // 解析单方 DPF 密钥，在 GPU 上执行求值，并将压缩输出拷回主机侧。
    template <typename T>
    std::vector<u32> evalDpf(Runtime &runtime, const KeyBlob &key, int party, const std::vector<T> &x, Stats *stats = nullptr)
    {
        if (x.empty())
            return {};

        T *d_x = detail::copyVectorToGpu(x);
        u8 *cursor = key.data();
        auto parsedKey = readGPUDPFKey(&cursor);
        u32 *d_out = gpuDpf(parsedKey, party, d_x, runtime.aes(), stats);
        gpuFree(d_x);
        return detail::copyPackedOutputToHost(d_out, parsedKey.memSzOut);
    }
}
}
