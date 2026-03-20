# DCF/DPF Minimal Runtime

这个仓库现在直接承载 DCF/DPF 的最小可运行集合，已经不再依赖 `GPU-MPC` 仓库中的其他源码目录。

## 文件

- `dpf.cu`: DPF 最小示例
- `dcf.cu`: DCF 最小示例
- `dpf_batch.cu`: DPF 运行时参数测试，输出 keygen/eval/transfer/total 耗时字段
- `dcf_batch.cu`: DCF 运行时参数测试，输出 keygen/eval/transfer/total 耗时字段
- `Makefile`: 当前根目录下的构建入口
- `工作进度.md`: 剥离计划、阶段进展、后续任务
- `fss/`: 当前已剥离进本目录的 FSS 相关代码
- `gpu/`: 当前已剥离进本目录的 GPU 基础设施代码
- `aes/`: 当前已剥离进本目录的 AES backend 代码
- `runtime/`: 独立运行时封装

## 构建

先设置环境变量：

```bash
export CUDA_VERSION=13.1
export GPU_ARCH=86
```

然后在仓库根目录执行：

```bash
make dpf CUDA_VERSION=$CUDA_VERSION GPU_ARCH=$GPU_ARCH
./dpf

make dcf CUDA_VERSION=$CUDA_VERSION GPU_ARCH=$GPU_ARCH
./dcf

make dpf_batch CUDA_VERSION=$CUDA_VERSION GPU_ARCH=$GPU_ARCH
./dpf_batch 64 10000000

make dcf_batch CUDA_VERSION=$CUDA_VERSION GPU_ARCH=$GPU_ARCH
./dcf_batch 64 1 10000000
```

当前构建已经压缩到只依赖：

- 当前仓库内源码
- CUDA 基础环境
- `nvcc`、`libcuda`、`libcudart`、`libcurand`

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
  - `gpu/packing_utils.h`
  - `aes/gpu_aes_shm.h`
  - `aes/gpu_aes_shm.cu`
  - `aes/gpu_aes_table.h`

## 输出字段

`dpf_batch` 输出：

- `bin`: 输入位宽，单位 `bit`
- `n`: 批大小，单位 `elem`
- `keygen`: 生成两方密钥耗时，单位 `us`
- `eval_p0`: P0 评估耗时，单位 `us`
- `eval_p1`: P1 评估耗时，单位 `us`
- `transfer_p0`: P0 结果拷回主机耗时，单位 `us`
- `transfer_p1`: P1 结果拷回主机耗时，单位 `us`
- `total`: 整体流程耗时，单位 `us`

`dcf_batch` 在上述字段基础上额外输出：

- `bout`: 输出位宽，单位 `bit`
