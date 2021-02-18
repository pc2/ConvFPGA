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

/**
 * \brief  print time taken for 3d fft execution and data transfer
 * \param  exec_time    : average time in seconds to execute a parallel 3d FFT
 * \param  gather_time  : average time in seconds to gather results to the master node after transformation
 * \param  flops        : fftw_flops 
 * \param  N1, N2, N3   : fft size
 * \param  nprocs       : number of processes used
 * \param  nthreads     : number of threads used
 * \param  iter         : number of iterations
 * \return true if successful, false otherwise
 */
bool print_results(double exec_time, double gather_time, double flops, unsigned N, unsigned nprocs, unsigned nthreads, unsigned iter, unsigned how_many);

/**
 * \brief  compute walltime in milliseconds
 * \retval time in milliseconds
 */
double getTimeinMilliSec();

/**
 * \brief Compute (K*L)%M 
 * \return output of the computation
 */
double moda(unsigned K, unsigned L, unsigned M);

#endif // HELPER_HPP
