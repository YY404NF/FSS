# Orca ( DPF / DCF )

本代码为论文项目提取出来的DPF/DCF函数，论文性能测试结果：

论文环境：

- Ubuntu 20.04
- NVIDIA RTX A6000 GPU
- CUDA 11.7
- CMake 3.27.2
- g++-9

参数：bin = 64, bout = 1, n = 10,000,000

| 模式 | Naive | AES | AES+LAYOUT | AES+LAYOUT+MEM |
| --- | --- | --- | --- | --- |
| Time (ms) | 3305 | 840 | 716 | 523 |
| Speedup | - | 3.9× | 4.6× | 6.3× |

## 依赖

- CUDA 基础环境
- `nvcc`、`libcuda`、`libcudart`、`libcurand`

## 环境配置

如果是在一台干净的 Ubuntu 22.04 / 24.04 `amd64` 服务器上，可以直接在项目目录执行：

```bash
chmod +x setup.sh
./setup.sh
```

脚本会自动检测 NVIDIA 驱动；如果没有检测到可用驱动，会自动安装。

如果你想手动控制这个行为，可以这样执行：

```bash
INSTALL_DRIVER=1 ./setup.sh   # 强制安装/重装驱动
INSTALL_DRIVER=0 ./setup.sh   # 跳过驱动安装
```

脚本执行完成后，加载环境变量：

```bash
source .env.cuda
```

### 使用 CMake 构建

在仓库根目录执行：

```bash
cmake -S . -B build \
  -DCUDA_VERSION=$CUDA_VERSION \
  -DGPU_ARCH=$GPU_ARCH

cmake --build build -j
```

单独运行：

```bash
./build/dpf
./build/dcf
./build/dpf_benchmark 64 10000000
./build/dcf_benchmark 64 1 10000000
```

## 文件

- `dpf.cu`: DPF 最小示例
- `dpf_benchmark.cu`: DPF 运行时基准测试

- `dcf.cu`: DCF 最小示例
- `dcf_benchmark.cu`: DCF 运行时基准测试

- `CMakeLists.txt`: 当前根目录下的构建入口

- `fss/`: FSS 相关代码
- `gpu/`: GPU 基础设施代码
- `aes/`: AES backend 代码
- `runtime/`: 独立运行时封装

## Benchmark 字段

`dpf_benchmark` 实际输出：

```text
DPF benchmark finished
  bin: <bin> bit
  n: <n> elem
  keygen: <time> us
  eval_p0: <time> us
  eval_p1: <time> us
  transfer_p0: <time> us
  transfer_p1: <time> us
  total: <time> us
```

- `bin`: 输入位宽，单位 `bit`
- `n`: 批大小，单位 `elem`
- `keygen`: 生成两方 DPF 密钥耗时，单位 `us`
- `eval_p0`: `SERVER0` 求值总耗时，单位 `us`
- `eval_p1`: `SERVER1` 求值总耗时，单位 `us`
- `transfer_p0`: `SERVER0` 结果拷回主机耗时，单位 `us`
- `transfer_p1`: `SERVER1` 结果拷回主机耗时，单位 `us`
- `total`: 整体流程总耗时，单位 `us`

`dcf_benchmark` 实际输出：

```text
DCF benchmark finished
  bin: <bin> bit
  bout: <bout> bit
  n: <n> elem
  keygen: <time> us
  eval_p0: <time> us
  eval_p1: <time> us
  transfer_p0: <time> us
  transfer_p1: <time> us
  total: <time> us
```

- `bin`: 输入位宽，单位 `bit`
- `bout`: 输出位宽，单位 `bit`
- `n`: 批大小，单位 `elem`
- `keygen`: 生成两方 DCF 密钥耗时，单位 `us`
- `eval_p0`: `SERVER0` 求值总耗时，单位 `us`
- `eval_p1`: `SERVER1` 求值总耗时，单位 `us`
- `transfer_p0`: `SERVER0` 结果拷回主机耗时，单位 `us`
- `transfer_p1`: `SERVER1` 结果拷回主机耗时，单位 `us`
- `total`: 整体流程总耗时，单位 `us`
