// Arjun Ramaswami
#include <iostream>
#include "helper.hpp"
#include "fft_conv3D.hpp"
extern "C"{
#include "convfpga/convfpga.h"
}

using namespace std;

int main(int argc, char* argv[]){

  //std::string inp_fname, out_fname;
  CONFIG conv_config;

  parse_args(argc, argv, conv_config);
  print_config(conv_config);

  const char* platform = "Intel(R) FPGA SDK for OpenCL(TM)";
  const bool use_svm = false;
  
  int isInit = fpga_initialize(platform, conv_config.path.c_str(), use_svm);
  if(isInit != 0){
    fprintf(stderr, "FPGA initialization error\n");
    return EXIT_FAILURE;
  }

  const unsigned num = conv_config.num;

  unsigned num_pts = num * num * num;
  size_t inp_sz = sizeof(float2) * num * num * num;
  float2 *filter = (float2*)fpgaf_complex_malloc(inp_sz);
  float2 *sig = (float2*)fpgaf_complex_malloc(inp_sz);
  float2 *out = (float2*)fpgaf_complex_malloc(inp_sz);

  bool status = fpgaf_create_data(filter, num_pts);
  if(!status){
    cerr << "Error in Data Creation" << endl;
    free(sig);
    free(filter);
    free(out);
    return EXIT_FAILURE;
  }
  status = fpgaf_create_data(sig, num_pts);
  if(!status){
    cerr << "Error in Data Creation" << endl;
    free(sig);
    free(filter);
    free(out);
    return EXIT_FAILURE;
  }

  double temp_timer = 0.0, total_api_time = 0.0;
  fpga_t timing;
  for(size_t i = 0; i < conv_config.iter; i++){
    temp_timer = getTimeinMilliSec();
    timing = fpgaf_conv3D(conv_config.num, sig, filter, out);
    total_api_time += getTimeinMilliSec() - temp_timer;

    if(!conv_config.noverify){
  #ifdef USE_FFTW
      status = fft_conv3D_cpu(conv_config, sig, filter, out);
      if(status){
        free(sig);
        free(filter);
        free(out);
      }
  #endif
    }

    if(timing.valid == 0){
      cerr << "Invalid execution, timing found to be 0";
      free(sig);
      free(filter);
      free(out);
      return EXIT_FAILURE;
    }
  }  // iter
  // destroy FFT input and output
  free(sig);
  free(filter);
  free(out);

  // destroy fpga state
  fpga_final();

  // Verify convolution with library
  //display_measures(total_api_time, avg_rd, avg_wr, avg_exec, avg_hw_rd, avg_hw_wr, avg_hw_exec, conv_config);
  return EXIT_SUCCESS;
}