// Arjun Ramaswami
#include <iostream>
#include "helper.hpp"
#include "fft_conv3D.hpp"
extern "C"{
#include "convfpga/convfpga.h"
}

using namespace std;

int main(int argc, char* argv[]){

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

  const char* platform;
  if(conv_config.emulate){
    platform = "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
  }
  else{
    platform = "Intel(R) FPGA SDK for OpenCL(TM)";
  }

  int isInit = fpga_initialize(platform, conv_config.path.c_str(), conv_config.usesvm);
  if(isInit != 0){
    cerr << "FPGA initialization error\n";
    return EXIT_FAILURE;
  }

  const unsigned num = conv_config.num;
  const unsigned batch = conv_config.batch;

  const size_t filter_numpts = num * num * num;
  const size_t filter_inp_sz = sizeof(float2) * num * num * num;

  const size_t sig_numpts = num * num * num * batch;
  const size_t sig_sz = sizeof(float2) * num * num * num * batch;

  float2 *filter = (float2*)fpgaf_complex_malloc(filter_inp_sz);
  float2 *sig = (float2*)fpgaf_complex_malloc(sig_sz);
  float2 *out = (float2*)fpgaf_complex_malloc(sig_sz);
  fpga_t runtime[conv_config.iter];

  try{
    create_data(filter, filter_numpts);
    create_data(sig, sig_numpts);

    for(unsigned i = 0; i < conv_config.iter; i++){
      cout << endl << i << ": Calculating Conv3D" << endl;

      if(conv_config.usesvm && conv_config.batch > 1)
        runtime[i] = fpgaf_conv3D_svm_batch(conv_config.num, sig, filter, out, batch);
      else if(conv_config.usesvm  && conv_config.batch == 1)
        runtime[i] = fpgaf_conv3D_svm(conv_config.num, sig, filter, out);
      else if(!conv_config.usesvm  && conv_config.batch > 1)
        throw "Non-SVM Batch not implemented for Convolution 3D";
      else
        runtime[i] = fpgaf_conv3D(conv_config.num, sig, filter, out);

      if(runtime[i].valid == false){ throw "FPGA execution found invalid";}

      if(!conv_config.noverify){
        if(!fft_conv3D_cpu_verify(conv_config, sig, filter, out)){
          char excp[80];
          snprintf(excp, 80, "Iter %u: FPGA result incorrect in comparison to CPU\n", i);
          throw runtime_error(excp);
        }
      }
    }
  }
  catch(const char* msg){
    cerr << msg << endl;
    fpga_final();
    free(sig);
    free(filter);
    free(out);
    return EXIT_FAILURE;
  }

  // destroy fpga state
  fpga_final();

  // Verify convolution with library
  disp_results(conv_config, runtime); 
  
  free(sig);
  free(filter);
  free(out);

  return EXIT_SUCCESS;
}