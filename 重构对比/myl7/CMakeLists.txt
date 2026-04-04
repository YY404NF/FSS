set(MYL7_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../myl7")
set(MYL7_INCLUDE "${MYL7_ROOT}/include")

function(add_myl7_generated_target target_name template_name impl_name scheme_header scheme_name prg_mul scheme_type build_alpha_body build_query_body expect_expr)
  set(IMPL_NAME "${impl_name}")
  set(SCHEME_HEADER "${scheme_header}")
  set(SCHEME_NAME "${scheme_name}")
  set(PRG_MUL "${prg_mul}")
  set(SCHEME_TYPE "${scheme_type}")
  set(BUILD_ALPHA_BODY "${build_alpha_body}")
  set(BUILD_QUERY_BODY "${build_query_body}")
  set(EXPECT_EXPR "${expect_expr}")

  set(source_path "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.cu")
  set(binary_path "${CMAKE_CURRENT_BINARY_DIR}/${target_name}")

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/${template_name}"
    "${source_path}"
    @ONLY)

  set(extra_flags "")
  set(extra_libs "")
  if(target_name MATCHES "^myl7_cpu_")
    list(APPEND extra_flags "-Xcompiler=-maes,-msse4.2")
    list(APPEND extra_libs "-lcrypto")
  endif()

  if(OpenMP_CXX_FOUND)
    list(APPEND extra_flags "-Xcompiler=${OpenMP_CXX_FLAGS}")
  endif()

  add_custom_command(
    OUTPUT "${binary_path}"
    COMMAND "${CMAKE_CUDA_COMPILER}"
      -O3
      -std=c++20
      "--generate-code=arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH}"
      "-I${MYL7_INCLUDE}"
      "-I${CMAKE_CURRENT_SOURCE_DIR}/../common"
      ${extra_flags}
      "${source_path}"
      -o "${binary_path}"
      ${extra_libs}
    DEPENDS
      "${source_path}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${template_name}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../common/compare_bench_format.h"
    VERBATIM
    COMMAND_EXPAND_LISTS)

  add_custom_target(${target_name} ALL DEPENDS "${binary_path}")
endfunction()

set(DPF_ALPHA_BODY "    return static_cast<InType>(10ULL + 2ULL * globalIdx);")
set(DPF_QUERY_BODY "    return globalIdx % 3 == 0 ? alpha : static_cast<InType>(alpha + 1);")
set(DPF_EXPECT_EXPR "x == alpha")

set(DCF_ALPHA_BODY "    return static_cast<InType>(20ULL + 2ULL * globalIdx);")
set(DCF_QUERY_BODY
"    if (globalIdx % 4 == 0) return alpha;\n    if (globalIdx % 4 == 1) return static_cast<InType>(alpha - 1);\n    return static_cast<InType>(alpha + 1);")
set(DCF_EXPECT_EXPR "x < alpha")

add_myl7_generated_target(
  myl7_gpu_dpf_bench
  myl7_gpu_bench_template.cu.in
  myl7_gpu
  dpf
  DPF
  2
  "fss::Dpf<kInBits, GroupType, PrgType, InType>"
  "${DPF_ALPHA_BODY}"
  "${DPF_QUERY_BODY}"
  "${DPF_EXPECT_EXPR}")

add_myl7_generated_target(
  myl7_gpu_dcf_bench
  myl7_gpu_bench_template.cu.in
  myl7_gpu
  dcf
  DCF
  4
  "fss::Dcf<kInBits, GroupType, PrgType, InType, fss::DcfPred::kLt>"
  "${DCF_ALPHA_BODY}"
  "${DCF_QUERY_BODY}"
  "${DCF_EXPECT_EXPR}")

add_myl7_generated_target(
  myl7_cpu_dpf_bench
  myl7_cpu_bench_template.cu.in
  myl7_cpu
  dpf
  DPF
  2
  "fss::Dpf<kInBits, GroupType, PrgType, InType>"
  "${DPF_ALPHA_BODY}"
  "${DPF_QUERY_BODY}"
  "${DPF_EXPECT_EXPR}")

add_myl7_generated_target(
  myl7_cpu_dcf_bench
  myl7_cpu_bench_template.cu.in
  myl7_cpu
  dcf
  DCF
  4
  "fss::Dcf<kInBits, GroupType, PrgType, InType, fss::DcfPred::kLt>"
  "${DCF_ALPHA_BODY}"
  "${DCF_QUERY_BODY}"
  "${DCF_EXPECT_EXPR}")
