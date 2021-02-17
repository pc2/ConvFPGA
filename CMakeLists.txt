# Author: Arjun Ramaswami
cmake_minimum_required(VERSION 3.16)
project(Convolution 
          VERSION 1.0
          DESCRIPTION "3D FFT based Convolution"
          LANGUAGES CXX C)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

message("-- Building external dependencies")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/extDep.cmake)

find_package(IntelFPGAOpenCL REQUIRED)
message("-- IntelFPGAOpenCL found")

# build source
message("-- Building host code")
add_subdirectory(src)

# build kernel
message("-- Building kernel code")
message("  -- Generating Custom Targets")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/genKernelTargets.cmake)
add_subdirectory(kernels)

# build tests
message("-- Building tests")
add_subdirectory(tests)
enable_testing()

