#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "aes/gpu_aes_shm.h"
#include "gpu/gpu_mem.h"
#include "gpu/gpu_random.h"

namespace gpu_mpc
{
namespace standalone
{
    namespace detail
    {
        // 当前实现用固定预算估算一批 keygen/eval 可以容纳多少元素，避免一次分配过大。
        constexpr std::size_t kOneGiB = std::size_t(1) << 30;
        constexpr std::size_t kDpfBatchBudgetBytes = 24 * kOneGiB;

        inline int roundDownToWarpMultiple(int value)
        {
            return value - (value % 32);
        }

        inline int dpfBatchSize(int bin)
        {
            const std::size_t bytesPerElement = std::size_t(bin - LOG_AES_BLOCK_LEN + 2) * sizeof(AESBlock);
            int batch = static_cast<int>(kDpfBatchBudgetBytes / bytesPerElement);
            batch = roundDownToWarpMultiple(batch);
            return std::max(batch, 32);
        }

        inline int dcfBatchSize(int bin, int bout)
        {
            const int elemsPerBlock = AES_BLOCK_LEN_IN_BITS / bout;
            const int newBin = bin - static_cast<int>(std::log2(elemsPerBlock));
            const std::size_t bytesPerElement = std::size_t(newBin + 2) * sizeof(AESBlock);
            int batch = static_cast<int>(kDpfBatchBudgetBytes / bytesPerElement);
            batch = roundDownToWarpMultiple(batch);
            return std::max(batch, 32);
        }

        inline std::size_t sstableKeyBytes(int bin, int n)
        {
            return 2 * sizeof(int) + std::size_t(n) * (std::size_t(1) << std::max(0, bin - 3));
        }

        inline std::size_t dpfTreeKeyBytes(int bin, int n, bool evalAll)
        {
            const std::size_t memSzScw = std::size_t(n) * std::size_t(bin - LOG_AES_BLOCK_LEN) * sizeof(AESBlock);
            const std::size_t memSzL = std::size_t(n) * sizeof(AESBlock);
            const std::size_t memSzT = evalAll
                ? std::size_t(n) * sizeof(u32)
                : (((std::size_t(n) - 1) / PACKING_SIZE) + 1) * sizeof(PACK_TYPE) * std::size_t(bin - LOG_AES_BLOCK_LEN);
            return 3 * sizeof(int) + memSzScw + (2 * memSzL) + memSzT;
        }

        inline std::size_t dpfKeyBytes(int bin, int n, bool evalAll)
        {
            if (bin <= 7)
                return sstableKeyBytes(bin, n);

            const int batch = dpfBatchSize(bin);
            const int blocks = (n - 1) / batch + 1;
            std::size_t total = 3 * sizeof(int);
            for (int b = 0; b < blocks; ++b)
            {
                total += dpfTreeKeyBytes(bin, std::min(batch, n - b * batch), evalAll);
            }
            return total;
        }

        inline std::size_t dcfTreeKeyBytes(int bin, int bout, int n)
        {
            const int elemsPerBlock = AES_BLOCK_LEN_IN_BITS / bout;
            const int newBin = bin - static_cast<int>(std::log2(elemsPerBlock));
            const std::size_t memSzK = std::size_t(n) * std::size_t(newBin) * sizeof(AESBlock);
            const std::size_t memSzL = 2 * std::size_t(n) * sizeof(AESBlock);
            const std::size_t memSzV = ((((std::size_t)bout * n) - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE) * std::size_t(newBin - 1);
            return 3 * sizeof(int) + memSzK + memSzL + memSzV;
        }

        inline std::size_t dcfKeyBytes(int bin, int bout, int n, bool leq, bool unitPayload)
        {
            if (bin <= 8)
            {
                if (!(bout == 1 && unitPayload && leq))
                    throw std::invalid_argument("Small-bin DCF only supports bout=1, payload=1, leq=true.");
                return sstableKeyBytes(bin, n);
            }

            const int batch = dcfBatchSize(bin, bout);
            const int blocks = (n - 1) / batch + 1;
            std::size_t total = 4 * sizeof(int);
            for (int b = 0; b < blocks; ++b)
            {
                total += dcfTreeKeyBytes(bin, bout, std::min(batch, n - b * batch));
            }
            return total;
        }

        template <typename T>
        inline T *copyVectorToGpu(const std::vector<T> &values)
        {
            if (values.empty())
                return nullptr;
            return reinterpret_cast<T *>(moveToGPU(reinterpret_cast<u8 *>(const_cast<T *>(values.data())),
                                                   values.size() * sizeof(T),
                                                   nullptr));
        }

        inline std::vector<u32> copyPackedOutputToHost(u32 *d_out, std::size_t bytes)
        {
            std::vector<u32> out(bytes / sizeof(u32));
            auto *h_out = reinterpret_cast<u32 *>(moveToCPU(reinterpret_cast<u8 *>(d_out), bytes, nullptr));
            std::memcpy(out.data(), h_out, bytes);
            cpuFree(h_out);
            gpuFree(d_out);
            return out;
        }
    }

    class Runtime
    {
    public:
        // 统一接管本地最小运行链需要的 GPU 内存池、AES 上下文和随机源。
        Runtime()
        {
            initGPUMemPool();
            initAESContext(&aes_);
            initGPURandomness();
        }

        // 当前最小集合只需要显式销毁 GPU 随机源，其余资源由底层模块自行托管。
        ~Runtime()
        {
            destroyGPURandomness();
        }

        Runtime(const Runtime &) = delete;
        Runtime &operator=(const Runtime &) = delete;

        AESGlobalContext *aes()
        {
            return &aes_;
        }

    private:
        AESGlobalContext aes_{};
    };

    class KeyBlob
    {
    public:
        // KeyBlob 负责承载序列化后的 key 字节流，供 dpf/dcf 两条 facade 共享。
        explicit KeyBlob(std::size_t capacityBytes)
            : data_(cpuMalloc(capacityBytes, true)), capacity_(capacityBytes), size_(0)
        {
        }

        ~KeyBlob()
        {
            if (data_ != nullptr)
                cpuFree(data_);
        }

        KeyBlob(const KeyBlob &) = delete;
        KeyBlob &operator=(const KeyBlob &) = delete;

        KeyBlob(KeyBlob &&other) noexcept
            : data_(other.data_), capacity_(other.capacity_), size_(other.size_)
        {
            other.data_ = nullptr;
            other.capacity_ = 0;
            other.size_ = 0;
        }

        KeyBlob &operator=(KeyBlob &&other) noexcept
        {
            if (this != &other)
            {
                if (data_ != nullptr)
                    cpuFree(data_);
                data_ = other.data_;
                capacity_ = other.capacity_;
                size_ = other.size_;
                other.data_ = nullptr;
                other.capacity_ = 0;
                other.size_ = 0;
            }
            return *this;
        }

        u8 *data() const
        {
            return reinterpret_cast<u8 *>(data_);
        }

        std::size_t size() const
        {
            return size_;
        }

        void setSize(std::size_t sizeBytes)
        {
            if (sizeBytes > capacity_)
                throw std::out_of_range("KeyBlob size exceeds allocated capacity.");
            size_ = sizeBytes;
        }

    private:
        void *data_;
        std::size_t capacity_;
        std::size_t size_;
    };

    inline std::vector<u64> unpackPackedOutput(const std::vector<u32> &packed, int n, int bout)
    {
        // DPF/DCF 的 GPU 输出是按位压缩的，这里负责还原为逐元素结果。
        std::vector<u64> unpacked(n);
        if (bout == 1 || bout == 2)
        {
            const std::size_t bitMask = (std::size_t(1) << bout) - 1;
            for (int i = 0; i < n; ++i)
            {
                const std::size_t bitOffset = std::size_t(i) * bout;
                const std::size_t wordIndex = bitOffset / 32;
                const int shift = static_cast<int>(bitOffset % 32);

                std::uint64_t value = packed[wordIndex] >> shift;
                if (shift + bout > 32 && wordIndex + 1 < packed.size())
                    value |= std::uint64_t(packed[wordIndex + 1]) << (32 - shift);

                unpacked[i] = value & bitMask;
            }
            return unpacked;
        }

        for (int i = 0; i < n; ++i)
        {
            u64 value = 0;
            std::memcpy(&value, packed.data() + (i * sizeof(u64) / sizeof(u32)), sizeof(u64));
            unpacked[i] = value;
        }
        return unpacked;
    }
}
}
