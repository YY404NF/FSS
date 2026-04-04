set(LIBFSS_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../libfss/cpp")
set(PATCHED_LIBFSS_DIR "${CMAKE_CURRENT_BINARY_DIR}/patched_libfss")

file(MAKE_DIRECTORY "${PATCHED_LIBFSS_DIR}")

file(READ "${LIBFSS_CPP_ROOT}/fss-common.h" LIBFSS_COMMON_H)
string(REPLACE
  "return (n & ( 1 << (64-pos))) >> (64-pos);"
  "return static_cast<int>((n >> (64 - pos)) & 1ULL);"
  LIBFSS_COMMON_H_PATCHED
  "${LIBFSS_COMMON_H}")

if(LIBFSS_COMMON_H_PATCHED STREQUAL LIBFSS_COMMON_H)
  message(FATAL_ERROR "Failed to patch libfss getBit(...) for the 64-bit path.")
endif()

file(WRITE "${PATCHED_LIBFSS_DIR}/fss-common.h" "${LIBFSS_COMMON_H_PATCHED}")
configure_file("${LIBFSS_CPP_ROOT}/fss-client.cpp" "${PATCHED_LIBFSS_DIR}/fss-client.cpp" COPYONLY)
configure_file("${LIBFSS_CPP_ROOT}/fss-client.h" "${PATCHED_LIBFSS_DIR}/fss-client.h" COPYONLY)
configure_file("${LIBFSS_CPP_ROOT}/fss-server.cpp" "${PATCHED_LIBFSS_DIR}/fss-server.cpp" COPYONLY)
configure_file("${LIBFSS_CPP_ROOT}/fss-server.h" "${PATCHED_LIBFSS_DIR}/fss-server.h" COPYONLY)
configure_file("${LIBFSS_CPP_ROOT}/fss-common.cpp" "${PATCHED_LIBFSS_DIR}/fss-common.cpp" COPYONLY)
configure_file("${LIBFSS_CPP_ROOT}/openssl-aes.h" "${PATCHED_LIBFSS_DIR}/openssl-aes.h" COPYONLY)

find_library(GMP_LIBRARY gmp REQUIRED)
find_library(GMPXX_LIBRARY gmpxx REQUIRED)

add_executable(libfss_compare_bench
  libfss_compare_bench.cpp
  "${PATCHED_LIBFSS_DIR}/fss-client.cpp"
  "${PATCHED_LIBFSS_DIR}/fss-server.cpp"
  "${PATCHED_LIBFSS_DIR}/fss-common.cpp")

target_include_directories(libfss_compare_bench PRIVATE
  "${PATCHED_LIBFSS_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../common")

target_compile_definitions(libfss_compare_bench PRIVATE
  AESNI
  OPENSSL_AES_H
  aesni_set_encrypt_key=AES_set_encrypt_key
  aesni_set_decrypt_key=AES_set_decrypt_key
  aesni_encrypt=AES_encrypt
  aesni_decrypt=AES_decrypt)

target_compile_options(libfss_compare_bench PRIVATE -O3 -maes -msse4.2)

target_link_libraries(libfss_compare_bench PRIVATE
  OpenSSL::Crypto
  OpenSSL::SSL
  Threads::Threads
  ${GMP_LIBRARY}
  ${GMPXX_LIBRARY}
  ${CMAKE_DL_LIBS})
