/**
 * @file fft_conv3D.hpp
 * @brief Header file with APIs for fft-based 3D convolution
 */

#ifndef FFT_CONV3D_HPP
#define FFT_CONV3D_HPP

extern "C"{
#include "convfpga/convfpga.h"
}
struct CONFIG{
  std::string path;
  std::string wisdomfile;
  unsigned num; 
  unsigned iter;
  unsigned threads;
  unsigned batch;
  bool noverify;
  bool cpuonly;
  bool usesvm;
  bool emulate;
};

typedef struct cpu_timing {
  double filter_t;    /**< Time for filter  */ 
  double conv_t;      /**< Time for Convolution */ 
  bool valid;         /**< True if valid execution */
} cpu_t;

cpu_t fft_conv3D_cpu(struct CONFIG& config);

bool fft_conv3D_cpu_verify(struct CONFIG& config, float2 *sig, float2 *filter, float2 *out);

#endif 