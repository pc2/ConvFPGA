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
  double filter_pcie_wr_t;/**< Time to write from DDR to host using PCIe bus */ 
  double filter_exec_t;   /**< Kernel execution time from CPU wall clock time */

  double sig_pcie_wr_t;   /**< Time to write from DDR to host using PCIe bus */ 
  double sig_pcie_rd_t;   /**< Time to read from DDR to host using PCIe bus  */ 
  double sig_exec_t;    /**< Kernel execution time from CPU wall clock time */
  double siginv_exec_t;  /**< Kernel execution time from CPU wall clock time */

  bool valid;             /**< Represents true signifying valid execution */
} fpga_t;

extern int fpga_initialize(const char *platform_name, const char *path, const bool use_svm);

extern void fpga_final();

extern void* fpgaf_complex_malloc(const size_t sz);

extern fpga_t fpgaf_conv3D(const unsigned N, const float2 *sig, const float2 *filter, float2 *out);

extern fpga_t fpgaf_conv3D_svm(const unsigned N, const float2 *sig, const float2 *filter, float2 *out);

extern fpga_t fpgaf_conv3D_svm_batch(const unsigned N, const float2 *sig, const float2 *filter, float2 *out, const unsigned how_many);
#endif
