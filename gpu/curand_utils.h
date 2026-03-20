/* CURAND 相关错误检查与辅助宏，供 GPU 随机数模块使用。 */
#pragma once

#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <curand.h>

// CUDA API error checking
#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);       \
      throw std::runtime_error("CUDA error");                                  \
    }                                                                          \
  } while (0)

// curand API error checking
#define CURAND_CHECK(err)                                                      \
  do {                                                                         \
    curandStatus_t err_ = (err);                                               \
    if (err_ != CURAND_STATUS_SUCCESS) {                                       \
      std::printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("curand error");                                \
    }                                                                          \
  } while (0)

template <typename T> void print_vector(const std::vector<T> &data);

template <> void print_vector(const std::vector<float> &data) {
  for (auto &i : data)
    std::printf("%0.6f\n", i);
}

template <> void print_vector(const std::vector<unsigned int> &data) {
  for (auto &i : data)
    std::printf("%d\n", i);
}