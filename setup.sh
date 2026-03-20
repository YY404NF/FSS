#!/usr/bin/env bash

# 严格模式
set -euo pipefail

# CUDA 版本，默认 13.1
CUDA_VERSION="${CUDA_VERSION:-13.1}"
CUDA_APT_VERSION="${CUDA_VERSION/./-}"
INSTALL_DRIVER="${INSTALL_DRIVER:-auto}"
ENV_FILE="${ENV_FILE:-.env.cuda}"

# 日志
log() {
  printf '[setup] %s\n' "$*"
}

warn() {
  printf '[setup] 警告: %s\n' "$*" >&2
}

die() {
  printf '[setup] 错误: %s\n' "$*" >&2
  exit 1
}

# 权限检查
if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  command -v sudo >/dev/null 2>&1 || die "请使用 root 运行，或先安装 sudo"
  SUDO="sudo"
fi

# apt-get 命令检查
command -v apt-get >/dev/null 2>&1 || die "当前脚本仅支持 Ubuntu / Debian 风格的 apt 环境"

# wget 命令检查
command -v wget >/dev/null 2>&1 || {
  log "先安装 wget"
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y wget
}

# 系统版本检查
if [[ ! -r /etc/os-release ]]; then
  die "无法读取 /etc/os-release"
fi

# shellcheck disable=SC1091
source /etc/os-release

# 当前脚本面向 Ubuntu 22.04 / 24.04
if [[ "${ID:-}" != "ubuntu" ]]; then
  die "不支持的发行版: ${ID:-unknown}；当前脚本面向 Ubuntu 22.04 / 24.04"
fi

# 生成 CUDA 仓库目录
case "${VERSION_ID:-}" in
  22.04)
    CUDA_DISTRO="ubuntu2204"
    ;;
  24.04)
    CUDA_DISTRO="ubuntu2404"
    ;;
  *)
    die "不支持的 Ubuntu 版本: ${VERSION_ID:-unknown}；仅支持 22.04 和 24.04"
    ;;
esac

# 架构检查
ARCH="$(dpkg --print-architecture)"
if [[ "${ARCH}" != "amd64" ]]; then
  die "不支持的架构: ${ARCH}；当前项目默认面向 amd64"
fi

# 生成 CUDA 软件源地址
CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/${ARCH}"
CUDA_KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
CUDA_KEYRING_URL="${CUDA_REPO_URL}/${CUDA_KEYRING_DEB}"

# 基础依赖
log "安装基础构建依赖"
${SUDO} apt-get update
${SUDO} apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  git \
  gnupg \
  make \
  pkg-config

# cuda-keyring 检查
if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
  log "为 ${CUDA_DISTRO} 添加 NVIDIA CUDA APT 软件源"
  TMP_DEB="/tmp/${CUDA_KEYRING_DEB}"
  wget -O "${TMP_DEB}" "${CUDA_KEYRING_URL}"
  ${SUDO} dpkg -i "${TMP_DEB}"
  rm -f "${TMP_DEB}"
else
  log "检测到已安装 cuda-keyring，复用现有 NVIDIA APT 软件源"
fi

${SUDO} apt-get update

# 驱动存在性检查
DRIVER_PRESENT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    DRIVER_PRESENT=1
  fi
fi

# auto: 自动检测，缺驱动时自动安装
# 1:    强制安装/重装驱动
# 0:    跳过驱动安装
install_nvidia_driver() {
  local kernel_release
  kernel_release="$(uname -r)"

  log "安装内核头文件 linux-headers-${kernel_release}"
  if ! ${SUDO} apt-get install -y "linux-headers-${kernel_release}"; then
    die "安装 linux-headers-${kernel_release} 失败；请检查当前内核是否有对应 headers 包，或切换到 Ubuntu 官方内核后重试"
  fi

  log "安装 NVIDIA 驱动包 cuda-drivers"
  ${SUDO} apt-get install -y cuda-drivers
}

case "${INSTALL_DRIVER}" in
  auto)
    if [[ "${DRIVER_PRESENT}" == "1" ]]; then
      log "检测到已有可用的 NVIDIA 驱动，跳过驱动安装"
    else
      log "未检测到可用 NVIDIA 驱动，自动安装 cuda-drivers"
      install_nvidia_driver
    fi
    ;;
  1)
    log "按要求强制安装 NVIDIA 驱动（cuda-drivers）"
    install_nvidia_driver
    ;;
  0)
    warn "由于 INSTALL_DRIVER=0，跳过驱动安装"
    ;;
  *)
    die "无效的 INSTALL_DRIVER=${INSTALL_DRIVER}；可选值为 auto、1 或 0"
    ;;
esac

# 二次驱动检测
DRIVER_PRESENT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    DRIVER_PRESENT=1
  fi
fi

# 驱动可用性检查
if [[ "${DRIVER_PRESENT}" == "0" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  warn "检测到 nvidia-smi 命令存在但暂不可用；如果刚安装完驱动，可能需要重启"
fi

# CUDA Toolkit
log "安装 CUDA Toolkit ${CUDA_VERSION}"
${SUDO} apt-get install -y "cuda-toolkit-${CUDA_APT_VERSION}"

# 确认 nvcc 可用
if [[ ! -x "/usr/local/cuda-${CUDA_VERSION}/bin/nvcc" ]]; then
  die "安装完成后仍未找到 /usr/local/cuda-${CUDA_VERSION}/bin/nvcc"
fi

# 生成 GPU_ARCH
GPU_ARCH_VALUE="${GPU_ARCH:-}"
if [[ -z "${GPU_ARCH_VALUE}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARCH_VALUE="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')"
fi
if [[ -z "${GPU_ARCH_VALUE}" ]]; then
  GPU_ARCH_VALUE="86"
  warn "自动探测 GPU 计算能力失败，回退到默认 GPU_ARCH=${GPU_ARCH_VALUE}"
fi

# 构建环境变量文件
cat > "${ENV_FILE}" <<EOF
export CUDA_VERSION=${CUDA_VERSION}
export GPU_ARCH=${GPU_ARCH_VALUE}
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:\${LD_LIBRARY_PATH:-}
EOF

log "已写入环境变量文件: $(pwd)/${ENV_FILE}"

# GPU 状态
if command -v nvidia-smi >/dev/null 2>&1; then
  log "检测到 nvidia-smi，当前 GPU 信息如下:"
  nvidia-smi || true
else
  warn "当前仍无法使用 nvidia-smi；如果刚安装完驱动，请先重启再编译"
fi

# 打印 nvcc 版本
log "nvcc 版本:"
"/usr/local/cuda-${CUDA_VERSION}/bin/nvcc" --version

# 提示
cat <<EOF

环境配置完成。

下一步:
  1. source ${ENV_FILE}
  2. make all CUDA_VERSION=\$CUDA_VERSION GPU_ARCH=\$GPU_ARCH

快速运行:
  source ${ENV_FILE}
  ./dpf
  ./dcf

说明:
  - 当前脚本面向 Ubuntu 22.04 / 24.04 amd64。
  - INSTALL_DRIVER 默认是 auto，会在缺少驱动时自动安装。
  - 可设置 INSTALL_DRIVER=1 强制安装，或 INSTALL_DRIVER=0 跳过驱动安装。
  - 如果驱动是刚安装的，请先重启，确保 libcuda 和 nvidia-smi 可用。
EOF
