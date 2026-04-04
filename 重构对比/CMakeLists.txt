cmake_minimum_required(VERSION 3.22)

set(GPU_ARCH "" CACHE STRING "Target NVIDIA GPU architecture, for example 86 or 89")

if(NOT GPU_ARCH)
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE GPU_ARCH_RAW
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(GPU_ARCH_RAW)
    string(REGEX REPLACE "\n.*" "" GPU_ARCH_RAW "${GPU_ARCH_RAW}")
    string(REPLACE "." "" GPU_ARCH "${GPU_ARCH_RAW}")
  endif()
endif()

if(NOT GPU_ARCH)
  set(GPU_ARCH "86")
  message(WARNING "Failed to detect GPU architecture automatically. Falling back to GPU_ARCH=${GPU_ARCH}.")
endif()

project(restructured_fss_compare LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "${GPU_ARCH}")

find_package(CUDAToolkit REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto SSL)
find_package(OpenMP)
find_package(Threads REQUIRED)

message(STATUS "Using GPU_ARCH=${GPU_ARCH}")
message(STATUS "Using CUDA compiler version=${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "Using CUDAToolkit version=${CUDAToolkit_VERSION}")

add_subdirectory(Orca)
add_subdirectory(myl7)
add_subdirectory(libfss)
