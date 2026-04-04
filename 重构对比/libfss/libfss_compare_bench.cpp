#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

#include "compare_bench_format.h"
#include "fss-client.h"
#include "fss-server.h"

namespace
{

using Clock = std::chrono::high_resolution_clock;

constexpr int kBin = 64;
constexpr int kBout = 1;

unsigned long long microsBetween(const Clock::time_point &start, const Clock::time_point &end)
{
    return static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

std::size_t floorPowerOfTwo(std::size_t value)
{
    std::size_t out = 1;
    while ((out << 1) <= value) out <<= 1;
    return out;
}

std::size_t pickDefaultChunk(std::string_view primitive)
{
    constexpr std::size_t kTargetBytes = 256ULL * 1024ULL * 1024ULL;
    const std::size_t per_item_key_bytes =
        primitive == "dpf"
            ? 2ULL * (sizeof(ServerKeyEq) + 2ULL * static_cast<std::size_t>(kBin - 1) * sizeof(CWEq))
            : 2ULL * (sizeof(ServerKeyLt) + 2ULL * static_cast<std::size_t>(kBin - 1) * sizeof(CWLt));
    const std::size_t rough = std::max<std::size_t>(1, kTargetBytes / std::max<std::size_t>(1, per_item_key_bytes));
    return std::max<std::size_t>(1, floorPowerOfTwo(rough));
}

uint64_t buildDpfAlpha(std::size_t globalIdx)
{
    return 10ULL + 2ULL * globalIdx;
}

uint64_t buildDpfQuery(std::size_t globalIdx, uint64_t alpha)
{
    return (globalIdx % 3 == 0) ? alpha : (alpha + 1);
}

uint64_t buildDcfAlpha(std::size_t globalIdx)
{
    return 20ULL + 2ULL * globalIdx;
}

uint64_t buildDcfQuery(std::size_t globalIdx, uint64_t alpha)
{
    if (globalIdx % 4 == 0) return alpha;
    if (globalIdx % 4 == 1) return alpha - 1;
    return alpha + 1;
}

void freeEqKeys(std::vector<ServerKeyEq> &k0, std::vector<ServerKeyEq> &k1)
{
    for (std::size_t i = 0; i < k0.size(); ++i)
    {
        free(k0[i].cw[0]);
        free(k0[i].cw[1]);
        free(k1[i].cw[0]);
        free(k1[i].cw[1]);
        k0[i].cw[0] = nullptr;
        k0[i].cw[1] = nullptr;
        k1[i].cw[0] = nullptr;
        k1[i].cw[1] = nullptr;
    }
}

void freeLtKeys(std::vector<ServerKeyLt> &k0, std::vector<ServerKeyLt> &k1)
{
    for (std::size_t i = 0; i < k0.size(); ++i)
    {
        free(k0[i].cw[0]);
        free(k0[i].cw[1]);
        free(k1[i].cw[0]);
        free(k1[i].cw[1]);
        k0[i].cw[0] = nullptr;
        k0[i].cw[1] = nullptr;
        k1[i].cw[0] = nullptr;
        k1[i].cw[1] = nullptr;
    }
}

int runDpf(std::size_t n, std::size_t chunk)
{
    Fss client{};
    Fss server{};
    initializeClient(&client, static_cast<uint32_t>(kBin), 2);
    initializeServer(&server, &client);

    unsigned long long keygen_us = 0;
    unsigned long long eval_p0_us = 0;
    unsigned long long eval_p1_us = 0;
    const auto total_start = Clock::now();

    for (std::size_t offset = 0; offset < n; offset += chunk)
    {
        const std::size_t cur = std::min(chunk, n - offset);
        std::vector<uint64_t> alphas(cur);
        std::vector<uint64_t> xs(cur);
        for (std::size_t i = 0; i < cur; ++i)
        {
            const std::size_t global_idx = offset + i;
            alphas[i] = buildDpfAlpha(global_idx);
            xs[i] = buildDpfQuery(global_idx, alphas[i]);
        }

        std::vector<ServerKeyEq> k0(cur);
        std::vector<ServerKeyEq> k1(cur);
        const std::size_t check_n = std::min<std::size_t>(16, cur);
        std::vector<mpz_class> y0_check(check_n);
        std::vector<mpz_class> y1_check(check_n);

        auto start = Clock::now();
        for (std::size_t i = 0; i < cur; ++i)
        {
            generateTreeEq(&client, &k0[i], &k1[i], alphas[i], 1);
        }
        keygen_us += microsBetween(start, Clock::now());

        start = Clock::now();
        for (std::size_t i = 0; i < cur; ++i)
        {
            const mpz_class y = evaluateEq(&server, &k0[i], xs[i]);
            if (i < check_n) y0_check[i] = y;
        }
        eval_p0_us += microsBetween(start, Clock::now());

        start = Clock::now();
        for (std::size_t i = 0; i < cur; ++i)
        {
            const mpz_class y = evaluateEq(&server, &k1[i], xs[i]);
            if (i < check_n) y1_check[i] = y;
        }
        eval_p1_us += microsBetween(start, Clock::now());

        for (std::size_t i = 0; i < check_n; ++i)
        {
            const bool hit = xs[i] == alphas[i];
            const mpz_class sum = y0_check[i] - y1_check[i];
            if ((hit && sum != 1) || (!hit && sum != 0))
            {
                std::fprintf(stderr, "libfss dpf verification failed at global_idx=%zu\n", offset + i);
                freeEqKeys(k0, k1);
                return 2;
            }
        }

        freeEqKeys(k0, k1);
    }

    CompareBenchReport report;
    report.impl = "libfss_cpu";
    report.primitive = "dpf";
    report.bin = kBin;
    report.bout = kBout;
    report.n = static_cast<unsigned long long>(n);
    report.chunk_n = static_cast<unsigned long long>(chunk);
    report.eval_iters = 1;
    report.keygen_us = static_cast<double>(keygen_us);
    report.setup_p0_us = 0.0;
    report.setup_p1_us = 0.0;
    report.eval_p0_us = static_cast<double>(eval_p0_us);
    report.eval_p1_us = static_cast<double>(eval_p1_us);
    report.transfer_p0_us = 0.0;
    report.transfer_p1_us = 0.0;
    report.total_us = static_cast<double>(microsBetween(total_start, Clock::now()));
    printCompareBenchReport(report);
    return 0;
}

int runDcf(std::size_t n, std::size_t chunk)
{
    Fss client{};
    Fss server{};
    initializeClient(&client, static_cast<uint32_t>(kBin), 2);
    initializeServer(&server, &client);

    unsigned long long keygen_us = 0;
    unsigned long long eval_p0_us = 0;
    unsigned long long eval_p1_us = 0;
    const auto total_start = Clock::now();

    for (std::size_t offset = 0; offset < n; offset += chunk)
    {
        const std::size_t cur = std::min(chunk, n - offset);
        std::vector<uint64_t> alphas(cur);
        std::vector<uint64_t> xs(cur);
        for (std::size_t i = 0; i < cur; ++i)
        {
            const std::size_t global_idx = offset + i;
            alphas[i] = buildDcfAlpha(global_idx);
            xs[i] = buildDcfQuery(global_idx, alphas[i]);
        }

        std::vector<ServerKeyLt> k0(cur);
        std::vector<ServerKeyLt> k1(cur);
        const std::size_t check_n = std::min<std::size_t>(16, cur);
        std::vector<uint64_t> y0_check(check_n);
        std::vector<uint64_t> y1_check(check_n);

        auto start = Clock::now();
        for (std::size_t i = 0; i < cur; ++i)
        {
            generateTreeLt(&client, &k0[i], &k1[i], alphas[i], 1);
        }
        keygen_us += microsBetween(start, Clock::now());

        start = Clock::now();
        for (std::size_t i = 0; i < cur; ++i)
        {
            const uint64_t y = evaluateLt(&server, &k0[i], xs[i]);
            if (i < check_n) y0_check[i] = y;
        }
        eval_p0_us += microsBetween(start, Clock::now());

        start = Clock::now();
        for (std::size_t i = 0; i < cur; ++i)
        {
            const uint64_t y = evaluateLt(&server, &k1[i], xs[i]);
            if (i < check_n) y1_check[i] = y;
        }
        eval_p1_us += microsBetween(start, Clock::now());

        for (std::size_t i = 0; i < check_n; ++i)
        {
            const bool hit = xs[i] < alphas[i];
            const uint64_t sum = y0_check[i] - y1_check[i];
            if ((hit && sum != 1) || (!hit && sum != 0))
            {
                std::fprintf(stderr, "libfss dcf verification failed at global_idx=%zu\n", offset + i);
                freeLtKeys(k0, k1);
                return 2;
            }
        }

        freeLtKeys(k0, k1);
    }

    CompareBenchReport report;
    report.impl = "libfss_cpu";
    report.primitive = "dcf";
    report.bin = kBin;
    report.bout = kBout;
    report.n = static_cast<unsigned long long>(n);
    report.chunk_n = static_cast<unsigned long long>(chunk);
    report.eval_iters = 1;
    report.keygen_us = static_cast<double>(keygen_us);
    report.setup_p0_us = 0.0;
    report.setup_p1_us = 0.0;
    report.eval_p0_us = static_cast<double>(eval_p0_us);
    report.eval_p1_us = static_cast<double>(eval_p1_us);
    report.transfer_p0_us = 0.0;
    report.transfer_p1_us = 0.0;
    report.total_us = static_cast<double>(microsBetween(total_start, Clock::now()));
    printCompareBenchReport(report);
    return 0;
}

void printUsage(const char *prog)
{
    std::fprintf(stderr, "Usage: %s dpf <n> [chunk_n]\n", prog);
    std::fprintf(stderr, "       %s dcf <n> [chunk_n]\n", prog);
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4)
    {
        printUsage(argv[0]);
        return 1;
    }

    const std::string_view primitive = argv[1];
    const std::size_t n = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    if (n == 0)
    {
        printUsage(argv[0]);
        return 1;
    }

    const std::size_t chunk =
        argc == 4 ? static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10)) : pickDefaultChunk(primitive);
    if (chunk == 0)
    {
        printUsage(argv[0]);
        return 1;
    }

    if (primitive == "dpf")
        return runDpf(n, chunk);
    if (primitive == "dcf")
        return runDcf(n, chunk);

    printUsage(argv[0]);
    return 1;
}
