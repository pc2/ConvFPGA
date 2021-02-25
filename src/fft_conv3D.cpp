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
  //const int *inembed = n, *onembed = n;

  int threads_ok = fftwf_init_threads(); 
  if(threads_ok == 0){
    throw "Something went wrong with Multithreaded FFTW! Exiting... \n";
  }
  
  fftwf_plan_with_nthreads((int)config.threads);
  unsigned fftw_plan = FFTW_PLAN;
  //bool using_wisdom = true, wis_export = true;

  int status = fftwf_import_wisdom_from_filename("wisdom.txt");
  //int status = fftw_import_wisdom_from_filename(config.wisdomfile.c_str());
  if(status == 0){
    cout << "Cannot import wisdom" << endl;
  }
  else{
    cout << "Importing wisdom" << endl;
    //fftw_plan = FFTW_WISDOM_ONLY | fftw_plan;
  }


  /*
  ofstream wisfile;
  wisfile.open(config.wisdomfile);
  if(wisfile.is_open()){
    // File exists and wisdom can be imported
    cout << "Importing Wisdom from path: " << config.wisdomfile << endl;
    // If wisdom cannot be imported, create wisdom in the same filename
    int status = fftw_import_wisdom_from_filename(config.wisdomfile.c_str());
    if(status != 0){ // use wisdom
      cout << "File Found" << endl;
      fftw_plan = FFTW_WISDOM_ONLY | fftw_plan;
      wisfile.close();
    }
    else{// File exists but without any content, export to file
      cout << "-- Wisdom not found in file!" << endl;
      using_wisdom = false;
      wis_export = true;
    }
  }
  else{
    // No such file exists
    cout << "No file found in path\n";
    using_wisdom = false;
    wis_export = false;
    wisfile.close();
  }
  */
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

  double plan_start = getTimeinMilliSec();
  plan_filter = fftwf_plan_many_dft(dim, n, 1, fftwf_filter, NULL, istride, idist, fftwf_filter, NULL, ostride, odist, FFTW_FORWARD, fftw_plan);

  plan_sig = fftwf_plan_many_dft(dim, n, 1, fftwf_sig, NULL, istride, idist, fftwf_sig, NULL, ostride, odist, FFTW_FORWARD, fftw_plan);

  plan_inv_sig = fftwf_plan_many_dft(dim, n, 1, fftwf_sig, NULL, istride, idist, fftwf_sig, NULL, ostride, odist, FFTW_BACKWARD, fftw_plan);
  double plan_time = getTimeinMilliSec() - plan_start;

  cout << "Time to Plan: " << plan_time << endl;

  if(status == 0){
    int resWisdom = fftwf_export_wisdom_to_filename("wisdom.txt"); 
    printf ("Wisdom export res = %d\n", resWisdom);
  }

  /*
  if(using_wisdom == false && wis_export == true){
    cout << "Exporting wisdom created to path: " << config.wisdomfile << endl;
    //cout << "Exporting wisdom created to path: " << config.wisdomfile << endl;
    //if(!fftw_export_wisdom_to_filename(wisfile)){
    if(fftw_export_wisdom_to_filename(config.wisdomfile.c_str()) == 0){
      cerr << "Could not export wisdom to path\n";
    }
    wisfile.close();
  }
  */

  double conv_start = 0.0, conv_stop = 0.0;
  double filter_start = 0.0, filter_stop = 0.0;
  cpu_t timing_cpu = {0.0, 0.0, 0};

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
    float2 temp;
    for(unsigned i = 0; i < num_pts; i++){
      temp.x = (fftwf_sig[i][0] * fftwf_filter[i][0]) - (fftwf_sig[i][1] * fftwf_filter[i][1]);
      temp.y = (fftwf_sig[i][0] * fftwf_filter[i][1]) + (fftwf_sig[i][1] * fftwf_filter[i][0]);

      fftwf_sig[i][0] = temp.x;
      fftwf_sig[i][1] = temp.y;
    }
    
    // Inverse Transformation
    fftwf_execute(plan_inv_sig);
    conv_stop = getTimeinMilliSec();

    timing_cpu.conv_t += (conv_stop - conv_start);
    fftwf_free(fftwf_sig);
    fftwf_free(fftwf_filter);
  }

  cleanup_plans();
  fftwf_cleanup_threads();

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
