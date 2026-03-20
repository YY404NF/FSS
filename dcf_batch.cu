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
    const T limit = (bin == 64) ? ~T(0) : (T(1) << bin);
    for (int i = 0; i < n; ++i)
        rin[i] = T(20) + T(2) * i;
    assert(rin.empty() || rin.back() < limit);
    return rin;
}

static std::vector<T> buildQueries(const std::vector<T> &rin)
{
    // 构造查询输入：覆盖等于、小于和大于阈值三类情况，便于做批量性能测试
    std::vector<T> x(rin.size());
    for (std::size_t i = 0; i < rin.size(); ++i)
    {
        if (i % 4 == 0)
            x[i] = rin[i];
        else if (i % 4 == 1)
            x[i] = rin[i] - 1;
        else
            x[i] = rin[i] + 1;
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
    auto x = buildQueries(rin);

    // 总耗时
    const auto totalStart = std::chrono::high_resolution_clock::now();

    // 两方 DCF key 一次性生成耗时
    const auto keygenStart = std::chrono::high_resolution_clock::now();
    auto [dcfKey0, dcfKey1] = gpu_mpc::standalone::generateDcfKeys(runtime, bin, bout, rin, T(1), true);
    const auto keygenEnd = std::chrono::high_resolution_clock::now();

    // 统计 SERVER0 求值阶段的总耗时与传输耗时
    Stats p0Stats;
    const auto evalP0Start = std::chrono::high_resolution_clock::now();
    auto dcfShare0 = gpu_mpc::standalone::unpackPackedOutput(
        gpu_mpc::standalone::evalDcf(runtime, dcfKey0, SERVER0, x, &p0Stats),
        n,
        bout);
    const auto evalP0End = std::chrono::high_resolution_clock::now();

    // 统计 SERVER1 求值阶段的总耗时与传输耗时
    Stats p1Stats;
    const auto evalP1Start = std::chrono::high_resolution_clock::now();
    auto dcfShare1 = gpu_mpc::standalone::unpackPackedOutput(
        gpu_mpc::standalone::evalDcf(runtime, dcfKey1, SERVER1, x, &p1Stats),
        n,
        bout);
    const auto evalP1End = std::chrono::high_resolution_clock::now();

    const auto totalEnd = std::chrono::high_resolution_clock::now();

    std::printf(
        "DCF batch finished\n"
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
