#include <iostream>
#include <fstream>
#include <omp.h>
#include <fftw3.h>
#include <math.h>

#include "fft_conv3D.hpp"
extern "C"{
  #include "convfpga/convfpga.h"
}
#include "helper.hpp"
#include "config.h"

using namespace std;

static fftwf_plan plan_filter, plan_sig, plan_inv_sig;

static void cleanup_plans(){
  fftwf_destroy_plan(plan_sig);
  fftwf_destroy_plan(plan_filter);
  fftwf_destroy_plan(plan_inv_sig);
}

static bool fpgaf_create_data(fftwf_complex *inp, const unsigned num_pts){

  if(inp == NULL || num_pts <= 0){
    return false;
  }

  for(size_t i = 0; i < num_pts; i++){
    inp[i][0] = (float)((float)rand() / (float)RAND_MAX);
    inp[i][1] = (float)((float)rand() / (float)RAND_MAX);
  }

  return true;
}

// Convolution implementation
cpu_t fft_conv3D_cpu(struct CONFIG& config){

  unsigned num = config.num;

  size_t num_pts = num * num * num;
  fftwf_complex *fftwf_sig = fftwf_alloc_complex(num_pts);
  fftwf_complex *fftwf_filter = fftwf_alloc_complex(num_pts);

  const int dim = 3;
  const int n[3] = {(int)num, (int)num, (int)num};
  int idist = num * num * num, odist = num * num * num;
  int istride = 1, ostride = 1;

  int threads_ok = fftwf_init_threads(); 
  if(threads_ok == 0){
    throw "Something went wrong with Multithreaded FFTW! Exiting... \n";
  }
  
  fftwf_plan_with_nthreads((int)config.threads);
  unsigned fftw_plan = FFTW_PLAN;

  cout << "-- Creating Plan" << endl;
  switch(fftw_plan){
    case FFTW_MEASURE:  cout << "-- FFTW Plan: Measure\n";
                        break;
    case FFTW_ESTIMATE: cout << "-- FFTW Plan: Estimate\n";
                        break;
    case FFTW_PATIENT:  cout << "-- FFTW Plan: Patient\n";
                        break;
    case FFTW_EXHAUSTIVE: cout << "-- FFTW Plan: Exhaustive\n";
                        break;
    default: throw "-- Incorrect plan set\n";
            break;
  }

  int wis_status = fftwf_import_wisdom_from_filename(config.wisdomfile.c_str());
  if(wis_status == 0){
    cout << "-- Cannot import wisdom from " << config.wisdomfile << endl;
  }
  else{
    cout << "-- Importing wisdom from " << config.wisdomfile << endl;
    fftw_plan = FFTW_WISDOM_ONLY | FFTW_ESTIMATE;
  }

  double plan_start = getTimeinMilliSec();
  plan_filter = fftwf_plan_many_dft(dim, n, 1, fftwf_filter, NULL, istride, idist, fftwf_filter, NULL, ostride, odist, FFTW_FORWARD, fftw_plan);

  plan_sig = fftwf_plan_many_dft(dim, n, 1, fftwf_sig, NULL, istride, idist, fftwf_sig, NULL, ostride, odist, FFTW_FORWARD, fftw_plan);

  plan_inv_sig = fftwf_plan_many_dft(dim, n, 1, fftwf_sig, NULL, istride, idist, fftwf_sig, NULL, ostride, odist, FFTW_BACKWARD, fftw_plan);
  double plan_time = getTimeinMilliSec() - plan_start;

  cout << "-- Time to Plan: " << plan_time << endl;

  if(wis_status == 0){
    int exp_stat = fftwf_export_wisdom_to_filename(config.wisdomfile.c_str()); 
    if(exp_stat == 0){
      cout << "-- Could not export wisdom file to " << config.wisdomfile.c_str() << endl;
    }
    else{
      cout << "-- Exporting wisdom file to " << config.wisdomfile.c_str() << endl;
    }
  }

  double conv_start = 0.0, conv_stop = 0.0;
  double filter_start = 0.0, filter_stop = 0.0;
  cpu_t timing_cpu = {0.0, 0.0, false};

  for(unsigned i = 0; i < config.iter; i++){
    bool status = fpgaf_create_data(fftwf_filter, num_pts);
    if(!status){
      cerr << "Error in Data Creation" << endl;
      fftwf_free(fftwf_sig);
      fftwf_free(fftwf_filter);
      timing_cpu.valid = false;
      return timing_cpu;
    }
    status = fpgaf_create_data(fftwf_sig, num_pts);
    if(!status){
      cerr << "Error in Data Creation" << endl;
      fftwf_free(fftwf_sig);
      fftwf_free(fftwf_filter);
      timing_cpu.valid = false;
      return timing_cpu;
    }

    // Filter transformation
    filter_start = getTimeinMilliSec();
    fftwf_execute(plan_filter);
    filter_stop = getTimeinMilliSec();
    
    timing_cpu.filter_t += (filter_stop - filter_start);

    // Signal Transformation
    conv_start = getTimeinMilliSec();
    fftwf_execute(plan_sig);

    // Multiplication
    #pragma omp parallel for num_threads(config.threads)
    for(unsigned i = 0; i < num_pts; i++){
      float2 temp;
      temp.x = (fftwf_sig[i][0] * fftwf_filter[i][0]) - (fftwf_sig[i][1] * fftwf_filter[i][1]);
      temp.y = (fftwf_sig[i][0] * fftwf_filter[i][1]) + (fftwf_sig[i][1] * fftwf_filter[i][0]);

      fftwf_sig[i][0] = temp.x;
      fftwf_sig[i][1] = temp.y;
    }
    
    // Inverse Transformation
    fftwf_execute(plan_inv_sig);
    conv_stop = getTimeinMilliSec();

    timing_cpu.conv_t += (conv_stop - conv_start);

    cout << "Iter: " << i << endl;
    cout << "  Filter Exec: " << (filter_stop - filter_start);
    cout << "  Conv3D Exec: " << (conv_stop - conv_start);
    cout << endl;
  }

  timing_cpu.filter_t = timing_cpu.filter_t / config.iter;
  timing_cpu.conv_t = timing_cpu.conv_t / config.iter;

  cleanup_plans();
  fftwf_cleanup_threads();
  fftwf_free(fftwf_sig);
  fftwf_free(fftwf_filter);

  timing_cpu.valid = true;
  return timing_cpu;
}

// Verification Function for FPGA 3D Convolution
bool fft_conv3D_cpu_verify(struct CONFIG& config, const float2 *sig, const float2 *filter, float2 *fpgaout){

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

  /*
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
  */

  plan_filter = fftwf_plan_many_dft(dim, n, 1, fftwf_filter, NULL, istride, idist, fftwf_filter, NULL, ostride, odist, FFTW_FORWARD, fftw_plan);

  plan_sig = fftwf_plan_many_dft(dim, n, 1, fftwf_sig, NULL, istride, idist, fftwf_sig, NULL, ostride, odist, FFTW_FORWARD, fftw_plan);

  plan_inv_sig = fftwf_plan_many_dft(dim, n, 1, fftwf_sig, NULL, istride, idist, fftwf_sig, NULL, ostride, odist, FFTW_BACKWARD, fftw_plan);

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

  if(!config.cpuonly){
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
    float db = 10 * log(mag_sum / noise_sum) / log(10.0);
    if(db > 120){
      return true;
    }
    else{
      cout << "Signal to noise ratio on output sample: ";
      cout << db << " --> FAILED\n\n";
      return false;
    }
  } // end of if condition

  fftwf_free(fftwf_sig);
  fftwf_free(fftwf_filter);
  cleanup_plans();

  return true;
}
