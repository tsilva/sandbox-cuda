cmake_minimum_required(VERSION 3.16)
project(sandbox_cuda CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(multiply_benchmark src/multiply_benchmark.cu)
set_target_properties(multiply_benchmark PROPERTIES CUDA_ARCHITECTURES "52;72")
