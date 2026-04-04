#pragma once

#include <cstdint>
#include <cstdio>

struct CompareBenchReport
{
    const char *impl = "";
    const char *primitive = "";
    int bin = 0;
    int bout = 0;
    unsigned long long n = 0;
    unsigned long long chunk_n = 0;
    int eval_iters = 1;
    double keygen_us = 0.0;
    double setup_p0_us = 0.0;
    double setup_p1_us = 0.0;
    double eval_p0_us = 0.0;
    double eval_p1_us = 0.0;
    double transfer_p0_us = 0.0;
    double transfer_p1_us = 0.0;
    double total_us = 0.0;
};

inline void printCompareBenchReport(const CompareBenchReport &report)
{
    std::printf(
        "benchmark finished\n"
        "  impl: %s\n"
        "  primitive: %s\n"
        "  bin: %d bit\n"
        "  bout: %d bit\n"
        "  n: %llu elem\n"
        "  chunk_n: %llu elem\n"
        "  eval_iters: %d\n"
        "  keygen: %.2f us\n"
        "  setup_p0: %.2f us\n"
        "  setup_p1: %.2f us\n"
        "  eval_p0: %.2f us\n"
        "  eval_p1: %.2f us\n"
        "  transfer_p0: %.2f us\n"
        "  transfer_p1: %.2f us\n"
        "  total: %.2f us\n",
        report.impl,
        report.primitive,
        report.bin,
        report.bout,
        report.n,
        report.chunk_n,
        report.eval_iters,
        report.keygen_us,
        report.setup_p0_us,
        report.setup_p1_us,
        report.eval_p0_us,
        report.eval_p1_us,
        report.transfer_p0_us,
        report.transfer_p1_us,
        report.total_us);
}
