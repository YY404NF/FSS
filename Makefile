CUDA_VERSION ?= 13.1
GPU_ARCH ?= 86

CXX := /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc
FLAGS := -O3 -gencode arch=compute_$(GPU_ARCH),code=[sm_$(GPU_ARCH),compute_$(GPU_ARCH)] -std=c++17 -m64 -Xcompiler="-O3,-w,-std=c++17,-fpermissive,-fpic,-pthread,-fopenmp,-march=native"
LIBS := -lcuda -lcudart -lcurand
UTIL_FILES := $(CURDIR)/gpu/gpu_mem.cu
INCLUDES := -I '$(CURDIR)'

.PHONY: all dcf dpf dcf_benchmark dpf_benchmark clean

all: dpf dcf dpf_benchmark dcf_benchmark

dpf: dpf.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o dpf

dcf: dcf.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o dcf

dpf_benchmark: dpf_benchmark.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o dpf_benchmark

dcf_benchmark: dcf_benchmark.cu
	$(CXX) $(FLAGS) $(INCLUDES) $^ $(UTIL_FILES) $(LIBS) -o dcf_benchmark

clean:
	rm -f dpf dcf dpf_benchmark dcf_benchmark dpf_batch dcf_batch
