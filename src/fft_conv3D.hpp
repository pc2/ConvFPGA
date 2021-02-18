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
  std::string out_fname;
  unsigned num; 
  unsigned iter;
  unsigned threads;
  bool noverify;
};

bool fft_conv3D_cpu(struct CONFIG& config, const float2 *sig, const float2 *filter, float2 *out);

#endif 