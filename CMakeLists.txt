# Author: Arjun Ramaswami
cmake_minimum_required(VERSION 3.16)
project(Convolution 
          VERSION 1.0
          DESCRIPTION "3D FFT based Convolution"
          LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

message("-- Building external dependencies")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/extDep.cmake)

find_package(IntelFPGAOpenCL REQUIRED)
message("-- IntelFPGAOpenCL found")

# Find FFTW
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(FFTW REQUIRED)
message("-- FFTW found")

# Find OpenMP or MPI
find_package(MPI REQUIRED)

find_package(OpenMP REQUIRED)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/conv_config.h.in"
  "${CMAKE_BINARY_DIR}/include/conv_config.h"
  ESCAPE_QUOTES
)

# build host code
message("-- Building host code")
add_subdirectory(host)

# build kernel
message("-- Building kernel code")
message("  -- Generating Custom Targets")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/genKernelTargets.cmake)
add_subdirectory(kernels)

# build source code
message("-- Building source code")
add_subdirectory(src)

# build tests
message("-- Building tests")
add_subdirectory(tests)
enable_testing()

