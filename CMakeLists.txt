# Author: Arjun Ramaswami
cmake_minimum_required(VERSION 3.16)
project(Convolution 
          VERSION 1.0
          DESCRIPTION "3D FFT based Convolution"
          LANGUAGES CXX C)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

#configure_file(
#  "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in"
#  "${CMAKE_BINARY_DIR}/include/config.h"
#  ESCAPE_QUOTES
#)

# build source
add_subdirectory(src)

# build tests
add_subdirectory(tests)
enable_testing()