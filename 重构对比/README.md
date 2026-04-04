# 重构对比

这里整理了 3 个方向的统一测试入口：

- `Orca` 复用密钥版
- `myl7`
- `libfss`

当前只保留三边都方便直接对齐的公共口径：

- `DPF: bin=64`
- `DCF: bin=64, bout=1`

这样做的目的，是让别人只看 `/FSS` 目录时，也能直接编译并复现主要对比结果。

## 构建

在本目录执行：

```bash
cmake -S . -B build
cmake --build build -j
```

说明：

- `GPU_ARCH` 会优先通过 `nvidia-smi --query-gpu=compute_cap` 自动探测。
- 如果自动探测失败，会回退到 `86`。
- 如果本机 CUDA 不在默认环境里，可以手动传 `-DCUDAToolkit_ROOT=/usr/local/cuda-12.8` 之类的路径。

例如：

```bash
cmake -S . -B build -DGPU_ARCH=89
cmake --build build -j
```

## 可执行文件

### Orca

```bash
./build/Orca/orca_reuse_dpf_bench <n> [eval_iters]
./build/Orca/orca_reuse_dcf_bench <n> [eval_iters]
```

- 默认 `eval_iters=10`
- 这里的 `setup_p0/setup_p1` 表示两方 key 首次上传到 GPU 的时间

### myl7

```bash
./build/myl7/myl7_gpu_dpf_bench <n> [chunk_n]
./build/myl7/myl7_gpu_dcf_bench <n> [chunk_n]
./build/myl7/myl7_cpu_dpf_bench <n> [chunk_n]
./build/myl7/myl7_cpu_dcf_bench <n> [chunk_n]
```

- GPU 和 CPU 两边都固定到同一组参数
- `chunk_n` 不填时会自动按当前显存或内存估算

### libfss

```bash
./build/libfss/libfss_compare_bench dpf <n> [chunk_n]
./build/libfss/libfss_compare_bench dcf <n> [chunk_n]
```

- 当前固定 `bin=64`
- `DCF` 当前固定 `bout=1`
- `chunk_n` 不填时会自动使用较保守的默认分块

## 统一输出

三边 benchmark 都统一输出下面这些字段：

- `impl`
- `primitive`
- `bin`
- `bout`
- `n`
- `chunk_n`
- `eval_iters`
- `keygen`
- `setup_p0`
- `setup_p1`
- `eval_p0`
- `eval_p1`
- `transfer_p0`
- `transfer_p1`
- `total`

补充说明：

- `Orca` 的 `setup_p0/setup_p1` 是 key 首次上传时间
- `myl7` 和 `libfss` 当前没有单独拆这个阶段，所以这里记为 `0`
- `libfss` 是 CPU 实现，因此 `transfer_p0/transfer_p1` 也是 `0`
