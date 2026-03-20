#include <cassert>
#include <cstdio>
#include <vector>

#include "fss/dpf_api.h"

using T = u64;

int main()
{
    // Runtime 统一负责初始化 GPU 内存池、AES 上下文和随机源。
    gpu_mpc::standalone::Runtime runtime;

    // rin 表示每个位置对应的目标点；x 表示实际要查询的输入。
    // 这一版单例入口只保留一个很小的固定样例，用来验证 facade 是否可用。
    const std::vector<T> rin = {5, 12, 9, 3};
    const std::vector<T> x = {2, 12, 11, 1};

    // 生成两方 DPF 密钥。两把 key 分别交给 SERVER0 和 SERVER1 独立求值。
    auto [dpfKey0, dpfKey1] = gpu_mpc::standalone::generateDpfKeys(runtime, 8, rin);

    // 单独执行两方求值，再把按位打包输出还原成逐元素结果。
    auto dpfShare0 = gpu_mpc::standalone::unpackPackedOutput(
        gpu_mpc::standalone::evalDpf(runtime, dpfKey0, SERVER0, x),
        static_cast<int>(x.size()),
        1);
    auto dpfShare1 = gpu_mpc::standalone::unpackPackedOutput(
        gpu_mpc::standalone::evalDpf(runtime, dpfKey1, SERVER1, x),
        static_cast<int>(x.size()),
        1);

    // DPF 的最终结果通过两方 share 合并得到；这里验证语义是否等于 x[i] == rin[i]。
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const auto pointValue = (dpfShare0[i] + dpfShare1[i]) & 1ULL;
        const auto expected = static_cast<u64>(x[i] == rin[i]);
        std::printf("DPF[%zu] = %llu (expected %llu)\n", i, pointValue, expected);
        assert(pointValue == expected);
    }

    std::puts("Standalone DPF facade passed.");
    return 0;
}
