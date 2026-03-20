# FSS ( DPF / DCF )

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

## 构建

先设置环境变量：

```bash
export CUDA_VERSION=13.1
export GPU_ARCH=86
```

然后在仓库根目录执行：

```bash
make all
```

单独编译与运行：

```bash
make dpf CUDA_VERSION=$CUDA_VERSION GPU_ARCH=$GPU_ARCH
./dpf

make dcf CUDA_VERSION=$CUDA_VERSION GPU_ARCH=$GPU_ARCH
./dcf

make dpf_benchmark CUDA_VERSION=$CUDA_VERSION GPU_ARCH=$GPU_ARCH
./dpf_benchmark 64 10000000

make dcf_benchmark CUDA_VERSION=$CUDA_VERSION GPU_ARCH=$GPU_ARCH
./dcf_benchmark 64 1 10000000
```

## 文件

- `dpf.cu`: DPF 最小示例
- `dpf_benchmark.cu`: DPF 运行时基准测试

- `dcf.cu`: DCF 最小示例
- `dcf_benchmark.cu`: DCF 运行时基准测试

- `Makefile`: 当前根目录下的构建入口

- `fss/`: FSS 相关代码
- `gpu/`: GPU 基础设施代码
- `aes/`: AES backend 代码
- `runtime/`: 独立运行时封装

## Benchmark 字段

`dpf_benchmark`：

- `bin`: 输入位宽，单位 `bit`
- `n`: 批大小，单位 `elem`

`dcf_benchmark` 

- `bin`: 输入位宽，单位 `bit`
- `bout`: 输出位宽，单位 `bit`
- `n`: 批大小，单位 `elem`

通用字段：

- `keygen`: 生成两方密钥耗时，单位 `us`
- `eval_p0`: P0 评估耗时，单位 `us`
- `eval_p1`: P1 评估耗时，单位 `us`
- `transfer_p0`: P0 结果拷回主机耗时，单位 `us`
- `transfer_p1`: P1 结果拷回主机耗时，单位 `us`
- `total`: 整体流程耗时，单位 `us`
