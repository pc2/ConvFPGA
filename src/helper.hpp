// Author: Arjun Ramaswami
#ifndef HELPER_HPP
#define HELPER_HPP

extern "C"{
#include "convfpga/convfpga.h"
}
#include "fft_conv3D.hpp"

bool fpgaf_create_data(float2 *inp, unsigned N);

void parse_args(int argc, char* argv[], CONFIG &config);

void print_config(CONFIG config);

void disp_results(CONFIG config, fpga_t fpga_timing, double api_t);

void disp_results(CONFIG config, cpu_t cpu_timing);

/**
 * \brief  compute walltime in milliseconds
 * \retval time in milliseconds
 */
double getTimeinMilliSec();

#endif // HELPER_HPP
