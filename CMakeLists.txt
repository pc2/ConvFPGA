# Author: Arjun Ramaswami
cmake_minimum_required(VERSION 3.16)
project(convolution 
          VERSION 1.0
          DESCRIPTION "3D FFT based Convolution"
          LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# build external dependencies
message("-- Building external dependencies")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/extDep.cmake)

## find Intel FPGA SDK for OpenCL
find_package(IntelFPGAOpenCL REQUIRED)
message("-- IntelFPGAOpenCL found")

# Find FFTW
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(FFTW REQUIRED)
message("-- FFTW found")

find_package(OpenMP REQUIRED)

# build host code
message("-- Building host code")
add_subdirectory(host)

# build kernel
message("-- Building kernel code")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/genKernelTargets.cmake)
add_subdirectory(kernels)

# build source code
message("-- Building source code")
add_subdirectory(src)
