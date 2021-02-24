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

  if(conv_config.cpuonly){
    #ifdef USE_FFTW
    cpu_t cpu_timing = {0.0, 0.0, false};
    cpu_timing = fft_conv3D_cpu(conv_config);
    if(cpu_timing.valid == false){
      cout << "Error in CPU Conv3D Implementation\n";
      return EXIT_FAILURE;
    }
  
    disp_results(conv_config, cpu_timing); 
    return EXIT_SUCCESS;
    #else
    cerr << "FFTW not found" << endl;
    return EXIT_FAILURE;
    #endif
  }

  const unsigned num = conv_config.num;
  const unsigned num_pts = num * num * num;
  const size_t inp_sz = sizeof(float2) * num * num * num;

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

  const char* platform = "Intel(R) FPGA SDK for OpenCL(TM)";
  
  int isInit = fpga_initialize(platform, conv_config.path.c_str(), conv_config.usesvm);
  if(isInit != 0){
    cerr << "FPGA initialization error\n";
    return EXIT_FAILURE;
  }

  fpga_t timing_fpga;
  double temp_timer = 0.0, total_api_time = 0.0;
  for(size_t i = 0; i < conv_config.iter; i++){

    if(conv_config.usesvm){
      temp_timer = getTimeinMilliSec();
      timing_fpga = fpgaf_conv3D_svm(conv_config.num, sig, filter, out);
      total_api_time += getTimeinMilliSec() - temp_timer;
    }
    else{
      temp_timer = getTimeinMilliSec();
      timing_fpga = fpgaf_conv3D(conv_config.num, sig, filter, out);
      total_api_time += getTimeinMilliSec() - temp_timer;
    }
    if(!conv_config.noverify){
  #ifdef USE_FFTW
      status = fft_conv3D_cpu_verify(conv_config, sig, filter, out);
      if(!status){
        free(sig);
        free(filter);
        free(out);
      }
  #endif
    }

    if(timing_fpga.valid == false){
      cerr << "Invalid execution, timing found to be 0";
      free(sig);
      free(filter);
      free(out);
      return EXIT_FAILURE;
    }
  }  // iter
  double timing_api = total_api_time / conv_config.iter;

  // destroy FFT input and output
  free(sig);
  free(filter);
  free(out);

  // destroy fpga state
  fpga_final();

  // Verify convolution with library
  disp_results(conv_config, timing_fpga, timing_api); 
  
  return EXIT_SUCCESS;
}