#pragma once

#include <stdexcept>
#include <vector>

#include "runtime/standalone_runtime.h"
#include "fss/gpu_dcf.h"

class Stats;

namespace gpu_mpc
{
namespace standalone
{
    // 生成 DCF 两方密钥，并将结果封装为可独立保存/传递的二进制 blob。
    template <typename T>
    std::pair<KeyBlob, KeyBlob> generateDcfKeys(Runtime &runtime, int bin, int bout, const std::vector<T> &rin, T payload = T(1), bool leq = true)
    {
        if (rin.empty())
            throw std::invalid_argument("rin must not be empty.");

        const std::size_t keyBytes = detail::dcfKeyBytes(bin, bout, static_cast<int>(rin.size()), leq, payload == T(1));
        KeyBlob key0(keyBytes);
        KeyBlob key1(keyBytes);

        T *d_rin = detail::copyVectorToGpu(rin);

        u8 *cursor0 = key0.data();
        dcf::gpuKeyGenDCF(&cursor0, SERVER0, bin, bout, static_cast<int>(rin.size()), d_rin, payload, runtime.aes(), leq);
        key0.setSize(static_cast<std::size_t>(cursor0 - key0.data()));

        u8 *cursor1 = key1.data();
        dcf::gpuKeyGenDCF(&cursor1, SERVER1, bin, bout, static_cast<int>(rin.size()), d_rin, payload, runtime.aes(), leq);
        key1.setSize(static_cast<std::size_t>(cursor1 - key1.data()));

        gpuFree(d_rin);
        return {std::move(key0), std::move(key1)};
    }

    // 解析单方 DCF 密钥，在 GPU 上执行求值，并将压缩输出拷回主机侧。
    template <typename T>
    std::vector<u32> evalDcf(Runtime &runtime, const KeyBlob &key, int party, const std::vector<T> &x, Stats *stats = nullptr)
    {
        if (x.empty())
            return {};

        T *d_x = detail::copyVectorToGpu(x);
        u8 *cursor = key.data();
        auto parsedKey = dcf::readGPUDCFKey(&cursor);
        u32 *d_out = dcf::gpuDcf<T, 1, dcf::idPrologue, dcf::idEpilogue>(parsedKey, party, d_x, runtime.aes(), stats);
        gpuFree(d_x);
        return detail::copyPackedOutputToHost(d_out, parsedKey.memSzOut);
    }
}
}
