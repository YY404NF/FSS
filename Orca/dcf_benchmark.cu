#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fss/dcf_api.h"
#include "gpu/gpu_stats.h"

using T = u64;

static void printUsage(const char *prog)
{
    std::fprintf(stderr, "Usage: %s <bin> <bout> <n>\n", prog);
}

static std::vector<T> buildRin(int bin, int n)
{
    // 构造一组稳定、可复现的阈值输入，避免每次运行都依赖额外随机源
    std::vector<T> rin(n);
    if (bin == 64)
    {
        for (int i = 0; i < n; ++i)
            rin[i] = T(20) + T(2) * i;
        return rin;
    }

    const T limit = T(1) << bin;
    const T span = limit - 1;
    // 保证阈值始终落在 [1, 2^bin - 1]，避免后续构造 x = rin - 1 时发生域外值。
    constexpr T kStride = 104729;
    for (int i = 0; i < n; ++i)
        rin[i] = T(1) + ((T(19) + T(i) * kStride) % span);
    assert(rin.empty() || rin.back() < limit);
    return rin;
}

static std::vector<T> buildQueries(int bin, const std::vector<T> &rin)
{
    // 构造查询输入：覆盖等于、小于和大于阈值三类情况，便于做批量性能测试
    std::vector<T> x(rin.size());
    const T limit = (bin == 64) ? ~T(0) : (T(1) << bin);
    for (std::size_t i = 0; i < rin.size(); ++i)
    {
        if (i % 4 == 0)
            x[i] = rin[i];
        else if (i % 4 == 1)
            x[i] = rin[i] - 1;
        else
            x[i] = (rin[i] + 1 < limit) ? (rin[i] + 1) : rin[i];
    }
    return x;
}

static unsigned long long microsBetween(
    const std::chrono::high_resolution_clock::time_point &start,
    const std::chrono::high_resolution_clock::time_point &end)
{
    return static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printUsage(argv[0]);
        return 1;
    }

    const int bin = std::atoi(argv[1]);
    const int bout = std::atoi(argv[2]);
    const int n = std::atoi(argv[3]);
    if (bin <= 0 || bin > 64 || bout <= 0 || bout > 64 || n <= 0)
    {
        printUsage(argv[0]);
        return 1;
    }

    gpu_mpc::standalone::Runtime runtime;
    auto rin = buildRin(bin, n);
    auto x = buildQueries(bin, rin);

    // 总耗时
    const auto totalStart = std::chrono::high_resolution_clock::now();

    // 两方 DCF key 一次性生成耗时
    const auto keygenStart = std::chrono::high_resolution_clock::now();
    auto [dcfKey0, dcfKey1] = gpu_mpc::standalone::generateDcfKeys(runtime, bin, bout, rin, T(1), true);
    const auto keygenEnd = std::chrono::high_resolution_clock::now();

    // eval_p0/eval_p1 只统计 gpuDcf 本体调用时间
    T *d_x = gpu_mpc::standalone::detail::copyVectorToGpu(x);

    u8 *cursor0 = dcfKey0.data();
    auto parsedKey0 = dcf::readGPUDCFKey(&cursor0);

    // 统计 SERVER0 求值阶段的总耗时与传输耗时
    Stats p0Stats;
    const auto evalP0Start = std::chrono::high_resolution_clock::now();
    u32 *d_out0 = dcf::gpuDcf<T, 1, dcf::idPrologue, dcf::idEpilogue>(parsedKey0, SERVER0, d_x, runtime.aes(), &p0Stats);
    const auto evalP0End = std::chrono::high_resolution_clock::now();
    [[maybe_unused]] auto dcfShare0 = gpu_mpc::standalone::unpackPackedOutput(
        gpu_mpc::standalone::detail::copyPackedOutputToHost(d_out0, parsedKey0.memSzOut),
        n,
        bout);

    u8 *cursor1 = dcfKey1.data();
    auto parsedKey1 = dcf::readGPUDCFKey(&cursor1);

    // 统计 SERVER1 求值阶段的总耗时与传输耗时
    Stats p1Stats;
    const auto evalP1Start = std::chrono::high_resolution_clock::now();
    u32 *d_out1 = dcf::gpuDcf<T, 1, dcf::idPrologue, dcf::idEpilogue>(parsedKey1, SERVER1, d_x, runtime.aes(), &p1Stats);
    const auto evalP1End = std::chrono::high_resolution_clock::now();
    [[maybe_unused]] auto dcfShare1 = gpu_mpc::standalone::unpackPackedOutput(
        gpu_mpc::standalone::detail::copyPackedOutputToHost(d_out1, parsedKey1.memSzOut),
        n,
        bout);

    gpuFree(d_x);

    const auto totalEnd = std::chrono::high_resolution_clock::now();

    std::printf(
        "DCF benchmark finished\n"
        "  bin: %d bit\n"
        "  bout: %d bit\n"
        "  n: %d elem\n"
        "  keygen: %llu us\n"
        "  eval_p0: %llu us\n"
        "  eval_p1: %llu us\n"
        "  transfer_p0: %llu us\n"
        "  transfer_p1: %llu us\n"
        "  total: %llu us\n",
        bin,
        bout,
        n,
        microsBetween(keygenStart, keygenEnd),
        microsBetween(evalP0Start, evalP0End),
        microsBetween(evalP1Start, evalP1End),
        static_cast<unsigned long long>(p0Stats.transfer_time),
        static_cast<unsigned long long>(p1Stats.transfer_time),
        microsBetween(totalStart, totalEnd));

    return 0;
}
