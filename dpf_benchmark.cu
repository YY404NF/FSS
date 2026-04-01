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
    // 构造一组稳定、可复现的输入点，避免每次运行都依赖额外随机源
    std::vector<T> rin(n);
    if (bin == 64)
    {
        for (int i = 0; i < n; ++i)
            rin[i] = T(10) + T(2) * i;
        return rin;
    }

    const T limit = T(1) << bin;
    const T mask = limit - 1;
    // 用奇数步长在 2^bin 域内做稳定遍历，避免大批量时线性递增越界。
    constexpr T kStride = 104729;
    for (int i = 0; i < n; ++i)
        rin[i] = (T(10) + T(i) * kStride) & mask;
    assert(rin.empty() || rin.back() < limit);
    return rin;
}

static std::vector<T> buildQueries(int bin, const std::vector<T> &rin)
{
    // 构造查询输入：一部分位置命中目标点，其余位置偏移 1，便于覆盖命中/未命中两类情况
    std::vector<T> x(rin.size());
    const T limit = (bin == 64) ? ~T(0) : (T(1) << bin);
    for (std::size_t i = 0; i < rin.size(); ++i)
        x[i] = (i % 3 == 0 || rin[i] + 1 >= limit) ? rin[i] : (rin[i] + 1);
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
    auto x = buildQueries(bin, rin);

    // 总耗时
    const auto totalStart = std::chrono::high_resolution_clock::now();

    // 两方 DPF key 一次性生成耗时
    const auto keygenStart = std::chrono::high_resolution_clock::now();
    auto [dpfKey0, dpfKey1] = gpu_mpc::standalone::generateDpfKeys(runtime, bin, rin);
    const auto keygenEnd = std::chrono::high_resolution_clock::now();

    // eval_p0/eval_p1 只统计 gpuDpf 本体调用时间
    T *d_x = gpu_mpc::standalone::detail::copyVectorToGpu(x);

    u8 *cursor0 = dpfKey0.data();
    auto parsedKey0 = readGPUDPFKey(&cursor0);

    // 统计 SERVER0 求值阶段的总耗时与传输耗时
    Stats p0Stats;
    const auto evalP0Start = std::chrono::high_resolution_clock::now();
    u32 *d_out0 = gpuDpf(parsedKey0, SERVER0, d_x, runtime.aes(), &p0Stats);
    const auto evalP0End = std::chrono::high_resolution_clock::now();
    [[maybe_unused]] auto dpfShare0 = gpu_mpc::standalone::unpackPackedOutput(
        gpu_mpc::standalone::detail::copyPackedOutputToHost(d_out0, parsedKey0.memSzOut),
        n,
        1);

    u8 *cursor1 = dpfKey1.data();
    auto parsedKey1 = readGPUDPFKey(&cursor1);

    // 统计 SERVER1 求值阶段的总耗时与传输耗时
    Stats p1Stats;
    const auto evalP1Start = std::chrono::high_resolution_clock::now();
    u32 *d_out1 = gpuDpf(parsedKey1, SERVER1, d_x, runtime.aes(), &p1Stats);
    const auto evalP1End = std::chrono::high_resolution_clock::now();
    [[maybe_unused]] auto dpfShare1 = gpu_mpc::standalone::unpackPackedOutput(
        gpu_mpc::standalone::detail::copyPackedOutputToHost(d_out1, parsedKey1.memSzOut),
        n,
        1);

    gpuFree(d_x);

    const auto totalEnd = std::chrono::high_resolution_clock::now();

    std::printf(
        "DPF benchmark finished\n"
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
