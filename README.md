# DCF/DPF Extraction Work

这个目录专门放 DCF/DPF 剥离实验代码，不改动原项目入口。

## 文件

- `dpf.cu`: DPF 最小示例
- `dcf.cu`: DCF 最小示例
- `dpf_batch.cu`: DPF 运行时参数测试，输出 keygen/eval/transfer/total 耗时字段
- `dcf_batch.cu`: DCF 运行时参数测试，输出 keygen/eval/transfer/total 耗时字段
- `Makefile`: 仅用于这个工作目录下的示例构建
- `工作进度.md`: 剥离计划、阶段进展、后续任务
- `fss/`: 当前已剥离进本目录的 FSS 相关代码
- `gpu/`: 当前已剥离进本目录的 GPU 基础设施代码
- `aes/`: 当前已剥离进本目录的 AES backend 代码

## 构建

先设置环境变量：

```bash
export CUDA_VERSION=11.7
export GPU_ARCH=86
```

然后在仓库根目录执行：

```bash
make -C 剥离工作 dpf
./剥离工作/dpf

make -C 剥离工作 dcf
./剥离工作/dcf

make -C 剥离工作 dpf_batch
./剥离工作/dpf_batch 64 10000000

make -C 剥离工作 dcf_batch
./剥离工作/dcf_batch 64 1 10000000
```

当前构建已经压缩到只依赖：

- `剥离工作/` 内源码
- CUDA 基础环境

当前工作目标仍然是继续清理 `fss/` 和 `gpu/` 中未使用的历史残留，进一步逼近最小可运行集合。

## 当前最小运行文件

当前 `dpf/dcf` 构建链实际使用的文件主要分为 4 组：

- 入口：
  - `dpf.cu`
  - `dcf.cu`
- 运行时胶水：
  - `runtime/standalone_runtime.h`
  - `fss/dpf_api.h`
  - `fss/dcf_api.h`
- FSS 核心：
  - `fss/gpu_dpf.h`
  - `fss/gpu_dpf.cu`
  - `fss/gpu_dpf_templates.h`
  - `fss/gpu_dcf.h`
  - `fss/gpu_dcf.cu`
  - `fss/gpu_dcf_templates.h`
  - `fss/gpu_dcf_sstab.h`
  - `fss/gpu_sstab.h`
  - `fss/gpu_fss_helper.h`
- GPU/AES 基础设施：
  - `gpu/gpu_mem.h`
  - `gpu/gpu_mem.cu`
  - `gpu/gpu_random.h`
  - `gpu/gpu_random.cu`
  - `gpu/gpu_data_types.h`
  - `gpu/gpu_stats.h`
  - `gpu/helper_cuda.h`
  - `gpu/helper_string.h`
  - `gpu/curand_utils.h`
  - `gpu/misc_utils.h`
  - `aes/gpu_aes_shm.h`
  - `aes/gpu_aes_shm.cu`
  - `aes/gpu_aes_table.h`
