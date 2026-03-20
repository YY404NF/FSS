/* DPF 运行时参数批量入口：负责按 bin/n 生成测试数据并输出完整耗时字段。 */
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fss/dpf_api.h"
#include "gpu/gpu_stats.h"

using T = u64;

static void printUsage(const char *prog)
{
    std::fprintf(stderr, "Usage: %s <bin> <n>\n", prog);
}

static std::vector<T> buildRin(int bin, int n)
{
    // 构造一组稳定、可复现的输入点，避免每次运行都依赖额外随机源。
    std::vector<T> rin(n);
    const T limit = (bin == 64) ? ~T(0) : (T(1) << bin);
    for (int i = 0; i < n; ++i)
        rin[i] = T(10) + T(2) * i;
    assert(rin.empty() || rin.back() < limit);
    return rin;
}

static std::vector<T> buildQueries(const std::vector<T> &rin)
{
    // 构造查询输入：一部分位置命中目标点，其余位置偏移 1，便于覆盖命中/未命中两类情况。
    std::vector<T> x(rin.size());
    for (std::size_t i = 0; i < rin.size(); ++i)
        x[i] = (i % 3 == 0) ? rin[i] : (rin[i] + 1);
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
    if (argc != 3)
    {
        printUsage(argv[0]);
        return 1;
    }

    const int bin = std::atoi(argv[1]);
    const int n = std::atoi(argv[2]);
    if (bin <= 0 || bin > 64 || n <= 0)
    {
        printUsage(argv[0]);
        return 1;
    }

    gpu_mpc::standalone::Runtime runtime;
    auto rin = buildRin(bin, n);
    auto x = buildQueries(rin);

    // total_us 覆盖整次批量测试入口的总耗时。
    const auto totalStart = std::chrono::high_resolution_clock::now();

    // keygen_us 统计两方 DPF key 一次性生成耗时。
    const auto keygenStart = std::chrono::high_resolution_clock::now();
    auto [dpfKey0, dpfKey1] = gpu_mpc::standalone::generateDpfKeys(runtime, bin, rin);
    const auto keygenEnd = std::chrono::high_resolution_clock::now();

    // eval_p0_us / transfer_p0_us 统计 SERVER0 求值阶段的总耗时与传输耗时。
    Stats p0Stats;
    const auto evalP0Start = std::chrono::high_resolution_clock::now();
    auto dpfShare0 = gpu_mpc::standalone::unpackPackedOutput(
        gpu_mpc::standalone::evalDpf(runtime, dpfKey0, SERVER0, x, &p0Stats),
        n,
        1);
    const auto evalP0End = std::chrono::high_resolution_clock::now();

    // eval_p1_us / transfer_p1_us 统计 SERVER1 求值阶段的总耗时与传输耗时。
    Stats p1Stats;
    const auto evalP1Start = std::chrono::high_resolution_clock::now();
    auto dpfShare1 = gpu_mpc::standalone::unpackPackedOutput(
        gpu_mpc::standalone::evalDpf(runtime, dpfKey1, SERVER1, x, &p1Stats),
        n,
        1);
    const auto evalP1End = std::chrono::high_resolution_clock::now();

    const auto totalEnd = std::chrono::high_resolution_clock::now();

    // 这里不再打印样例输出，只输出性能字段，便于脚本化记录和后处理。
    std::printf(
        "DPF batch finished\n"
        "  bin: %d bit\n"
        "  n: %d elem\n"
        "  keygen: %llu us\n"
        "  eval_p0: %llu us\n"
        "  eval_p1: %llu us\n"
        "  transfer_p0: %llu us\n"
        "  transfer_p1: %llu us\n"
        "  total: %llu us\n",
        bin,
        n,
        microsBetween(keygenStart, keygenEnd),
        microsBetween(evalP0Start, evalP0End),
        microsBetween(evalP1Start, evalP1End),
        static_cast<unsigned long long>(p0Stats.transfer_time),
        static_cast<unsigned long long>(p1Stats.transfer_time),
        microsBetween(totalStart, totalEnd));

    return 0;
}
