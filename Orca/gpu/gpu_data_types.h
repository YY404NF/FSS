/* 本文件集中定义最小运行链共享的基础整数类型、AESBlock 以及若干全局常量。 */
// #ifndef GPU_DATA_TYPES_H
// #define GPU_DATA_TYPES_H

#pragma once

#include <utility>
#include <stdint.h>
#include <cstddef>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef int64_t i64;
typedef int32_t i32;

typedef unsigned __int128 AESBlock;

#define SERVER0 0
#define SERVER1 1
#define AES_BLOCK_LEN_IN_BITS 128
#define FULL_MASK 0xffffffff
#define LOG_AES_BLOCK_LEN 7

#define PACKING_SIZE 32
#define PACK_TYPE uint32_t

#define NUM_SHARED_MEM_BANKS 32

using orcaTemplateClass = u64;

namespace dcf
{
    namespace orca
    {
        namespace global
        {
            static const int bw = 64;
            static const int scale = 24;
        }
    }
}
