cmake_minimum_required(VERSION 3.10)

project(convKernels VERSION 1.0
            DESCRIPTION "Kernels for 3D FFT based convolution using FPGAs"
            LANGUAGES C CXX)

set(FPGA_BOARD_NAME $ENV{FPGA_BOARD_NAME} CACHE STRING "Target Board Name")
if(DEFINED FPGA_BOARD_NAME)
  message("-- FPGA Board Name: ${FPGA_BOARD_NAME}")
else()
  message(ERROR, "No Target board found")
endif()  

set(SDK_VERSION $ENV{QUARTUS_VERSION} CACHE STRING "SDK Version")
if(SDK_VERSION)
  message("-- SDK Version: ${SDK_VERSION}")
else()
  message(ERROR, "No SDK Version Found")
endif()

set(BSP_VERSION $ENV{QUARTUS_VERSION_BSP} CACHE STRING "BSP Version")
if(BSP_VERSION)
  message("-- BSP Version: ${BSP_VERSION}")
else()
  message(ERROR, "No BSP Found")
endif()

set(LOG_POINTS 3 CACHE STRING "Log of per sample data points")
math(EXPR POINTS "1 << ${LOG_POINTS}")

set(LOG_FFT_SIZE 6 CACHE STRING "Log of points of FFT")
set_property(CACHE LOG_FFT_SIZE PROPERTY STRINGS 4 5 6 7 8 9)
math(EXPR FFT_SIZE "1 << ${LOG_FFT_SIZE}")
message("-- FFT size is ${FFT_SIZE}")
math(EXPR DEPTH "1 << (${LOG_FFT_SIZE} + ${LOG_FFT_SIZE} - ${LOG_POINTS})")
message("-- DEPTH: ${DEPTH}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/common/conv_config.h.in"
  "${CMAKE_BINARY_DIR}/kernels/common/conv_config.h"
  ESCAPE_QUOTES
)

set(BURST_INTERLEAVING CACHE BOOL "Enable burst interleaving")
if(BURST_INTERLEAVING)
  set(INTERLEAVING "")
  set(BURST "burstinter")
  message("-- Burst interleaved global memory accesses")
else()
  set(INTERLEAVING "-no-interleaving=default")
  set(BURST "nointer")
  message("-- Non-interleaved global memory accesses")
endif()

if(${FPGA_BOARD_NAME} STREQUAL "pac_s10_usm")
  set(DDR_BUFFER_LOCATION "device" CACHE STRING "Buffer location of 3d Transpose")
  set(SVM_HOST_BUFFER_LOCATION "host" CACHE STRING "SVM host buffer location")
elseif(${FPGA_BOARD_NAME} STREQUAL "p520_hpc_sg280l")
  set(DDR_BUFFER_LOCATION "DDR" CACHE STRING "Buffer location of 3d Transpose")
  set(SVM_HOST_BUFFER_LOCATION "" CACHE STRING "SVM host buffer location")
else()
  set(DDR_BUFFER_LOCATION "" CACHE STRING "Buffer location of 3d Transpose")
  set(SVM_HOST_BUFFER_LOCATION "" CACHE STRING "SVM host buffer location")
endif()

set_property(CACHE DDR_BUFFER_LOCATION PROPERTY STRINGS "DDR" "device" "")
set_property(CACHE SVM_HOST_BUFFER_LOCATION PROPERTY STRINGS "" "host")

message("-- Buffer location for 3d Transpose: ${DDR_BUFFER_LOCATION}")
message("-- SVM host Buffer location: ${SVM_HOST_BUFFER_LOCATION}")

## Flags for different target options
set(AOC_FLAGS "-g -v ${INTERLEAVING} -board=${FPGA_BOARD_NAME}" CACHE STRING "AOC compiler flags")
separate_arguments(AOC_FLAGS)
set(EMU_FLAGS "-march=emulator" CACHE STRING "AOC emulation flags")
separate_arguments(EMU_FLAGS)
set(REP_FLAGS "-report -rtl" CACHE STRING "AOC report flags")
separate_arguments(REP_FLAGS)
set(PROF_FLAGS "-profile=all" CACHE STRING "AOC profile flags")
separate_arguments(PROF_FLAGS)

## 
# Call function to create custom build commands
# Generates targets:
#   - ${kernel_name}_emu: to generate emulation binary
#   - ${kernel_name}_rep: to generate report
#   - ${kernel_name}_syn: to generate synthesis binary
##
set(CL_PATH "${convKernels_SOURCE_DIR}")
set(kernels conv3D conv3D_svm)

if (INTELFPGAOPENCL_FOUND)
  gen_fft_targets(${kernels})
else()
  message("Intel FPGA OpenCL not found")
endif()
