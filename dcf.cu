#include <cassert>
#include <cstdio>
#include <vector>

#include "fss/dcf_api.h"

using T = u64;

int main()
{
    // Runtime 统一负责初始化 GPU 内存池、AES 上下文和随机源
    gpu_mpc::standalone::Runtime runtime;

    // rin 表示每个位置对应的阈值；x 表示实际要比较的输入
    // 这一版单例入口使用固定小样例，只验证 DCF facade 的基础功能是否正常
    const std::vector<T> rin = {5, 12, 9, 3};
    const std::vector<T> x = {2, 12, 11, 1};

    // 生成两方 DCF 密钥。这里配置为 bout=1、payload=1、leq=true
    // 因此最终语义是 x[i] <= rin[i]
    auto [dcfKey0, dcfKey1] = gpu_mpc::standalone::generateDcfKeys(runtime, 9, 1, rin, T(1), true);

    // 分别执行两方求值，再把打包输出还原成逐元素结果
    auto dcfShare0 = gpu_mpc::standalone::unpackPackedOutput(
        gpu_mpc::standalone::evalDcf(runtime, dcfKey0, SERVER0, x),
        static_cast<int>(x.size()),
        1);
    auto dcfShare1 = gpu_mpc::standalone::unpackPackedOutput(
        gpu_mpc::standalone::evalDcf(runtime, dcfKey1, SERVER1, x),
        static_cast<int>(x.size()),
        1);

    // DCF 的最终输出由两方 share 合并得到；这里验证语义是否等于 x[i] <= rin[i]
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const auto lessThanValue = (dcfShare0[i] + dcfShare1[i]) & 1ULL;
        const auto expected = static_cast<u64>(x[i] <= rin[i]);
        std::printf("DCF[%zu] = %llu (expected %llu)\n", i, lessThanValue, expected);
        assert(lessThanValue == expected);
    }

    std::puts("Standalone DCF facade passed.");
    return 0;
}
