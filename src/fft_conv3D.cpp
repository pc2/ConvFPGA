#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <fftw3.h>
#include <math.h>

#include "conv_config.h"
#include "fft_conv3D.hpp"
extern "C"{
  #include "convfpga/convfpga.h"
}
#include "helper.hpp"

using namespace std;

static fftwf_plan plan_filter, plan_sig, plan_inv_sig;

/**
 * \brief create single precision floating points values for FFT computation for each process level block
 * \param fftw_data   : pointer to 3d number of sp points for FFTW
 * \param verify_data : pointer to 3d number of sp points for verification
 * \param N           : number of points in each dimension
 * \param H1, H2, H3  : harmonic to modify frequency of discrete time signal
 * \param how_many    : number of batched implementations of FFTW
 */
void get_sp_many_input(fftwf_complex *sig, unsigned num, unsigned H1, unsigned H2, unsigned H3, unsigned how_many){

  unsigned index; 
  float TWOPI = 6.2831853071795864769;
  float phase, phase1, phase2, phase3;
  double re_val = 0.0, img_val = 0.0;
  unsigned S1 = num*num, S2 = num;

  for(size_t many = 0; many < how_many; many++){
    for(size_t i = 0; i < num; i++) {
      for(size_t j = 0; j < num; j++) {
        for(size_t k = 0; k < num; k++) {
          phase1 = moda(i, H1, num) / num;
          phase2 = moda(j, H2, num) / num;
          phase3 = moda(k, H3, num) / num;
          phase = phase1 + phase2 + phase3;

          index = (many * S1 * S2) + (i * S1) + (j * S2) + k;

          re_val = cosf( TWOPI * phase ) / (num * num * num);
          img_val = sinf( TWOPI * phase ) / (num * num * num);

          sig[index][0] = re_val;
          sig[index][1] = img_val;

  #ifdef DEBUG          
          printf(" %d %d %d : fftw[%d] = (%f, %f) \n", i, j, k, index, sig[index][0], sig[index][1]);
  #endif
        }
      }
    }
  }
}

static void cleanup(){
  // Cleanup : fftw data, plans and threads

  fftwf_destroy_plan(plan_sig);
  fftwf_destroy_plan(plan_filter);
  fftwf_destroy_plan(plan_inv_sig);
}

bool fft_conv3D_cpu(struct CONFIG& config, const float2 *sig, const float2 *filter, float2 *fpgaout){

  unsigned H1 = 1, H2 = 1, H3 = 1;
  cout << "num = " << config.num << endl;
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
  const int *inembed = n, *onembed = n;

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

  plan_filter = fftwf_plan_many_dft(dim, n, 1, fftwf_filter, inembed, istride, idist, fftwf_filter, onembed, ostride, odist, FFTW_FORWARD, fftw_plan);

  plan_sig = fftwf_plan_many_dft(dim, n, 1, fftwf_sig, inembed, istride, idist, fftwf_sig, onembed, ostride, odist, FFTW_FORWARD, fftw_plan);

  plan_inv_sig = fftwf_plan_many_dft(dim, n, 1, fftwf_sig, inembed, istride, idist, fftwf_sig, onembed, ostride, odist, FFTW_BACKWARD, fftw_plan);

  //double start = 0.0, stop = 0.0;
  // Iterate iter times
  get_sp_many_input(fftwf_sig, num, H1, H2, H3, 1);
  get_sp_many_input(fftwf_filter, num, H1, H2, H3, 1);

  //start = getTimeinMilliSec();
  fftwf_execute(plan_filter);
  fftwf_execute(plan_sig);

  for(size_t i = 0; i < data_sz; i++){
    fftwf_sig[i][0] = fftwf_sig[i][0] * fftwf_filter[i][0] - fftwf_sig[i][1] * fftwf_sig[i][1];
    fftwf_sig[i][1] = fftwf_sig[i][0] * fftwf_filter[i][1] + fftwf_sig[i][1] * fftwf_sig[i][0];
  }

  fftwf_execute(plan_inv_sig);
  //stop = getTimeinMilliSec();

  //double diff = stop - start;

  double magnitude, noise, mag_sum, noise_sum;
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
  // Calculate SNR
  float db = 10 * log(mag_sum / noise_sum) / log(10.0);

  // if SNR greater than 120, verification passes
  if(db > 120){
    return true;
  }
  else{
    cout << "Signal to noise ratio on output sample: ";
    cout << db << " --> FAILED\n\n";
    return false;
  }
}
