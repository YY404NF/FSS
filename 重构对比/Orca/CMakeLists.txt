set(ORCA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../Orca")

function(add_orca_reuse_target target_name source_name)
  add_executable(${target_name}
    ${source_name}
    gpu_mem_silent.cu)

  target_include_directories(${target_name} PRIVATE
    "${ORCA_ROOT}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../common")

  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-m64>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-w,-fpermissive,-fpic,-pthread,-march=native>)

  target_link_libraries(${target_name} PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::curand)

  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endfunction()

add_orca_reuse_target(orca_reuse_dpf_bench dpf_benchmark_reuse.cu)
add_orca_reuse_target(orca_reuse_dcf_bench dcf_benchmark_reuse.cu)
