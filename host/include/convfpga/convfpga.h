// Author: Arjun Ramaswami

/**
 * @file convfpga.h
 * @brief Header file that provides APIs for OpenCL Host code to compute  FFT-based Convolution in FPGA
 */

#ifndef CONVFPGA_H
#define CONVFPGA_H

#include <stdbool.h>

/**
 * Single Precision Complex Floating Point Data Structure
 */
typedef struct {
  float x; /**< real value */
  float y; /**< imaginary value */
} float2;

/**
 * Record time in milliseconds of different FPGA runtime stages
 */
typedef struct fpga_timing {
  double filter_pcie_rd_t;/**< Time to read from DDR to host using PCIe bus  */ 
  double filter_pcie_wr_t;/**< Time to write from DDR to host using PCIe bus */ 
  double filter_exec_t;   /**< Kernel execution time from CPU wall clock time */

  double sig_pcie_rd_t;   /**< Time to read from DDR to host using PCIe bus  */ 
  double sig_pcie_wr_t;   /**< Time to write from DDR to host using PCIe bus */ 
  double sig_exec_t;    /**< Kernel execution time from CPU wall clock time */
  double siginv_exec_t;  /**< Kernel execution time from CPU wall clock time */

  //double svm_copyin_t;    /**< Time to copy in data to SVM */
  //double svm_copyout_t;   /**< Time to copy data out of SVM */ 
  bool valid;             /**< Represents true signifying valid execution */
} fpga_t;

/** 
 * @brief Initialize FPGA
 * @param platform_name: name of the OpenCL platform
 * @param path         : path to binary
 * @param use_svm      : 1 if true 0 otherwise
 * @return 0 if successful 
          -1 Path to binary missing
          -2 Unable to find platform passed as argument
          -3 Unable to find devices for given OpenCL platform
          -4 Failed to create program, file not found in path
          -5 Device does not support required SVM
 */
extern int fpga_initialize(const char *platform_name, const char *path, bool use_svm);

/** 
 * @brief Release FPGA Resources
 */
extern void fpga_final();

/** 
 * @brief Allocate memory of single precision complex floating points
 * @param sz  : size_t : size to allocate
 * @return void ptr or NULL
 */
extern void* fpgaf_complex_malloc(const size_t sz);

/**
 * @brief  compute an out-of-place single precision complex 3D-FFT using the DDR of the FPGA
 * @return fpga_t : time taken in milliseconds for data transfers and execution
 */
extern fpga_t fpgaf_conv3D(unsigned N, float2 *sig, float2 *filter, float2 *out);

/**
 * @brief  compute an out-of-place single precision complex 3D-FFT using the DDR of the FPGA and Shared Virtual Memory for Host to Device Communication
 * @param  interleaving  : toggle interleaved device memory
 * @return fpga_t : time taken in milliseconds for data transfers and execution
 */
extern fpga_t fpgaf_conv3D_svm(unsigned N, float2 *sig, float2 *filter, float2 *out);

#endif
