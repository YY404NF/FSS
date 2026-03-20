#pragma once

#include "gpu_aes_table.h"

#define NUM_SHARED_MEM_BANKS 32

#define AES_128_ROUNDS 10
#define AES_128_ROUNDS_MIN_1 9

#define CYCLIC_ROT_RIGHT_1 0x4321
#define CYCLIC_ROT_RIGHT_2 0x5432
#define CYCLIC_ROT_RIGHT_3 0x6543

struct AESGlobalContext
{
	// 全局上下文保存 AES 查表实现需要的常驻 GPU 侧表。
	u32 *t0_g;
	u8 *Sbox_g;
	u32 *t4_0G, *t4_1G, *t4_2G, *t4_3G;
};

struct AESSharedContext
{
	// shared memory 上下文用于把全局表切到块内共享内存，减少查询开销。
	u32 (*t0_s)[NUM_SHARED_MEM_BANKS];
	u8 (*Sbox)[32][4];
	u32 *t4_0S;
	u32 *t4_1S;
	u32 *t4_2S;
	u32 *t4_3S;
};

#include "gpu_aes_shm.cu"
