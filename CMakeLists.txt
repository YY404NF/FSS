cmake_minimum_required(VERSION 3.24)

# 用户可通过 -DCUDA_VERSION=... 覆盖默认 CUDA 版本，用来定位 /usr/local/cuda-<version>。
set(CUDA_VERSION "13.1" CACHE STRING "CUDA toolkit version used to locate /usr/local/cuda-<version>")
# 用户可通过 -DGPU_ARCH=... 指定目标 GPU 架构，例如 86、89、90。
set(GPU_ARCH "86" CACHE STRING "Target NVIDIA GPU architecture, for example 86")

# 如果外部没有显式指定 CUDAToolkit_ROOT，就按本机约定的 CUDA 安装路径自动推导。
if(NOT DEFINED CUDAToolkit_ROOT)
  set(CUDAToolkit_ROOT "/usr/local/cuda-${CUDA_VERSION}" CACHE PATH "CUDA toolkit root directory")
endif()

# 声明项目名称，并开启 C++/CUDA 两种语言支持。
project(FSS LANGUAGES CXX CUDA)

# 统一要求主机侧与 CUDA 侧都使用 C++17/CUDA17 标准。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# 把用户传入的 GPU 架构直接交给 CMake，生成对应 sm_xx 的编译参数。
set(CMAKE_CUDA_ARCHITECTURES "${GPU_ARCH}")

# 查找 CUDA Toolkit；找不到时直接终止配置。
find_package(CUDAToolkit REQUIRED)
# OpenMP 在当前项目里主要沿用原 Makefile 的主机编译参数，能找到就一并链接。
find_package(OpenMP)

# 封装公共建目标逻辑：4 个可执行程序只是在入口 .cu 上不同，其余配置完全一致。
function(add_fss_executable target_name entry_source)
  add_executable(${target_name}
    ${entry_source}
    gpu/gpu_mem.cu
  )

  # include 路径
  target_include_directories(${target_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

  # 编译选项
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-m64>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-w,-fpermissive,-fpic,-pthread,-fopenmp,-march=native>
  )

  # 运行时依赖
  target_link_libraries(${target_name} PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::curand
  )

  # 如果系统检测到了 OpenMP，就把主机侧 OpenMP 运行时一并链接进目标。
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endfunction()

# 定义项目当前提供的 4 个可执行文件。
add_fss_executable(dpf dpf.cu)
add_fss_executable(dcf dcf.cu)
add_fss_executable(dpf_benchmark dpf_benchmark.cu)
add_fss_executable(dcf_benchmark dcf_benchmark.cu)
