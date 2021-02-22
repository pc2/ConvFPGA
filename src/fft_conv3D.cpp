#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <fftw3.h>
#include <math.h>

#include "fft_conv3D.hpp"
extern "C"{
  #include "convfpga/convfpga.h"
}
#include "helper.hpp"

using namespace std;

static fftwf_plan plan_filter, plan_sig, plan_inv_sig;

static void cleanup(){
  fftwf_destroy_plan(plan_sig);
  fftwf_destroy_plan(plan_filter);
  fftwf_destroy_plan(plan_inv_sig);
}

bool fft_conv3D_cpu(struct CONFIG& config, const float2 *sig, const float2 *filter, float2 *fpgaout, double &cpu_exec_t){

  unsigned num = config.num;

  size_t data_sz = num * num * num;
  fftwf_complex *fftwf_sig = fftwf_alloc_complex(data_sz);
  fftwf_complex *fftwf_filter = fftwf_alloc_complex(data_sz);

  for(size_t i = 0; i < data_sz; i++){
    fftwf_sig[i][0] = sig[i].x;
    fftwf_sig[i][1] = sig[i].y;

    fftwf_filter[i][0] = filter[i].x;
    fftwf_filter[i][1] = filter[i].y;
  }
  
  const int dim = 3;
  const int n[3] = {(int)num, (int)num, (int)num};
  int idist = num * num * num, odist = num * num * num;
  int istride = 1, ostride = 1;
  //const int *inembed = n, *onembed = n;

  const unsigned fftw_plan = FFTW_ESTIMATE;

  switch(fftw_plan){
    case FFTW_MEASURE:  cout << "FFTW Plan: Measure\n";
                        break;
    case FFTW_ESTIMATE: cout << "FFTW Plan: Estimate\n";
                        break;
    case FFTW_PATIENT:  cout << "FFTW Plan: Patient\n";
                        break;
    case FFTW_EXHAUSTIVE: cout << "FFTW Plan: Exhaustive\n";
                        break;
    default: throw "Incorrect plan\n";
            break;
  }

  plan_filter = fftwf_plan_many_dft(dim, n, 1, fftwf_filter, NULL, istride, idist, fftwf_filter, NULL, ostride, odist, FFTW_FORWARD, fftw_plan);

  plan_sig = fftwf_plan_many_dft(dim, n, 1, fftwf_sig, NULL, istride, idist, fftwf_sig, NULL, ostride, odist, FFTW_FORWARD, fftw_plan);

  plan_inv_sig = fftwf_plan_many_dft(dim, n, 1, fftwf_sig, NULL, istride, idist, fftwf_sig, NULL, ostride, odist, FFTW_BACKWARD, fftw_plan);

  double start = 0.0, stop = 0.0;
  start = getTimeinMilliSec();
  fftwf_execute(plan_filter);
  fftwf_execute(plan_sig);

  float2 temp;
  for(size_t i = 0; i < data_sz; i++){
    temp.x = (fftwf_sig[i][0] * fftwf_filter[i][0]) - (fftwf_sig[i][1] * fftwf_filter[i][1]);
    temp.y = (fftwf_sig[i][0] * fftwf_filter[i][1]) + (fftwf_sig[i][1] * fftwf_filter[i][0]);

    fftwf_sig[i][0] = temp.x;
    fftwf_sig[i][1] = temp.y;
  }
  
  fftwf_execute(plan_inv_sig);
  stop = getTimeinMilliSec();

  cpu_exec_t = stop - start;

  double magnitude = 0.0, noise = 0.0, mag_sum = 0.0, noise_sum = 0.0;
  for (size_t i = 0; i < data_sz; i++) {
    magnitude = fftwf_sig[i][0] * fftwf_sig[i][0] + \
                      fftwf_sig[i][1] * fftwf_sig[i][1];
    noise = (fftwf_sig[i][0] - fpgaout[i].x) \
        * (fftwf_sig[i][0] - fpgaout[i].x) + 
        (fftwf_sig[i][1] - fpgaout[i].y) * (fftwf_sig[i][1] - fpgaout[i].y);

    mag_sum += magnitude;
    noise_sum += noise;
#ifndef NDEBUG
    //printf("%zu : fpga - (%e %e) cpu - (%e %e)\n", i, fpgaout[i].x, fpgaout[i].y, fftwf_sig[i][0], fftwf_sig[i][1]);
#endif      
  }

  fftwf_free(fftwf_sig);
  fftwf_free(fftwf_filter);
  cleanup();

  float db = 10 * log(mag_sum / noise_sum) / log(10.0);
  if(db > 120){
    return true;
  }
  else{
    cout << "Signal to noise ratio on output sample: ";
    cout << db << " --> FAILED\n\n";
    cpu_exec_t = 0.0;
    return false;
  }
}
