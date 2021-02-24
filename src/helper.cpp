//  Author: Arjun Ramaswami
#include <iostream>
#include <iomanip>
#include "cxxopts.hpp"
#include "helper.hpp"

using namespace std;

/**
 * \brief  compute walltime in milliseconds
 * \retval time in milliseconds
 */
double getTimeinMilliSec(){
   struct timespec a;
   clock_gettime(CLOCK_MONOTONIC, &a);
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0E3;
}

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
bool print_results(double exec_time, double gather_time, double flops, unsigned N, unsigned nprocs, unsigned nthreads, unsigned iter, unsigned how_many){

  if(exec_time == 0.0)
    throw "Error in Run\n";
  
  double avg_exec = exec_time / iter;

  cout << "\nMeasurements\n" << "--------------------------\n";
  cout << "Processes           : " << nprocs << endl;
  cout << "Threads             : " << nthreads << endl;
  cout << "FFT Size            : " << N << "^3\n";
  cout << "Batch               : " << how_many << endl;
  cout << "Iterations          : " << iter << endl;
  cout << "Avg Tot Runtime     : " << setprecision(4) << avg_exec << " ms\n";
  cout << "Runtime per batch   : " << (avg_exec / how_many) << " ms\n";
  cout << "Throughput          : " << (flops * 1e-9) << " GFLOPs\n";
  cout << "Time to Transfer    : " << gather_time << "ms\n";
  cout << "--------------------------\n";

  return true;
}

void parse_args(int argc, char* argv[], CONFIG &config){

  try{
    cxxopts::Options options("Convolution3D", "3D Conv Filter on incoming images");
    options.add_options()
      ("p, path", "Path to bitstream", cxxopts::value<string>())
      ("o, out", "Filename to output results", cxxopts::value<string>()->default_value("a.out"))
      ("n, num", "Size of FFT dim", cxxopts::value<unsigned>()->default_value("64"))
      ("i, iter", "Number of iterations", cxxopts::value<unsigned>()->default_value("1"))
      ("t, threads", "Number of threads", cxxopts::value<unsigned>()->default_value("1"))
      ("y, noverify", "No verification", cxxopts::value<bool>()->default_value("false") )
      ("c, cpu-only", "CPU FFTW Only", cxxopts::value<bool>()->default_value("false") )
      ("s, usesvm", "SVM enabled", cxxopts::value<bool>()->default_value("false") )
      ("h,help", "Print usage")
    ;
    auto opt = options.parse(argc, argv);

    // print help
    if (opt.count("help")){
      cout << options.help() << endl;
      exit(0);
    }

    config.cpuonly = opt["cpu-only"].as<bool>();
    if(!config.cpuonly){
      if(opt.count("path")){
        config.path = opt["path"].as<string>();
      }
      else{
        cout << "\tPlease input path to bitstream" << endl;
        exit(1);
      }
    }
    if(opt.count("out")){
      config.out_fname = opt["out"].as<string>();
      cout << "Using default output filename - a.out" << endl;
    }

    config.num = opt["num"].as<unsigned>();
    config.threads = opt["threads"].as<unsigned>();
    config.iter = opt["iter"].as<unsigned>();
    config.noverify = opt["noverify"].as<bool>();
    config.usesvm = opt["usesvm"].as<bool>();
  }
  catch(const cxxopts::OptionException& e){
    cerr << "Error parsing options: " << e.what() << endl;
    exit(1);
  }
}

void print_config(CONFIG config){
  cout << endl;
  cout << "CONFIGURATION: \n";
  cout << "---------------\n";
  cout << "Bitstream    = " << config.path << endl;
  cout << "Points       = {"<< config.num << ", " << config.num << ", " << config.num << "}" << endl;
  #ifdef MEASURE
  cout << "FFTW Plan    = Measure     \n";
  #elif PATIENT
  cout << "FFTW Plan    = Patient     \n";
  #elif EXHAUSTIVE
  cout << "FFTW Plan    = Exhaustive  \n";
  #else
  cout << "FFTW Plan    = Estimate    \n";
  #endif
  cout << "Threads      = "<< config.threads << endl;
  cout << "Iterations   = "<< config.iter << endl;
  cout << "----------------\n\n";
}

/**
 * \brief  create random single precision complex floating point values  
 * \param  inp : pointer to float2 data of size N 
 * \param  N   : number of points in the array
 * \return true if successful
 */
bool fpgaf_create_data(float2 *inp, unsigned num_pts){

  if(inp == NULL || num_pts <= 0){
    return false;
  }

  for(size_t i = 0; i < num_pts; i++){
    inp[i].x = (float)((float)rand() / (float)RAND_MAX);
    inp[i].y = (float)((float)rand() / (float)RAND_MAX);
  }

  return true;
}

/**
 * \brief  compute walltime in milliseconds
 * \return time in milliseconds
 */
double getTimeinMilliseconds(){
   struct timespec a;
   if(clock_gettime(CLOCK_MONOTONIC, &a) != 0){
     fprintf(stderr, "Error in getting wall clock time \n");
     exit(EXIT_FAILURE);
   }
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0E3;
}

void disp_results(CONFIG config, fpga_t fpga_timing, double api_t){

  cout << endl << endl;
  cout << "MEASUREMENTS \n";
  cout << "--------------\n";
  cout << "Points                 : " << config.num << "^3\n";
  cout << "Iterations             : " << config.iter << endl << endl;
  cout << (config.usesvm ? "Using SVM\n":"");

  cout << "FPGA:" << endl;
  cout << "-----" << endl;
  cout << "- Filter:" << endl;
  cout << "  PCIe Host to Device : "<< fpga_timing.filter_pcie_wr_t << endl;
  cout << "  Execution           : "<< fpga_timing.filter_exec_t << endl;
  cout << "  PCIe Device to Host : "<< fpga_timing.filter_pcie_rd_t << endl;
  cout << endl;

  cout << "- Signal Convolution:" << endl;
  cout << "  PCIe Host to Device : "<< fpga_timing.sig_pcie_wr_t << endl;
  cout << "  FFT + Conv          : "<< fpga_timing.sig_exec_t << endl;
  cout << "  Inverse FFT         : "<< fpga_timing.siginv_exec_t << endl;
  cout << "  Total Computation   : "<< fpga_timing.siginv_exec_t + fpga_timing.siginv_exec_t << endl;
  cout << "  PCIe Device to Host : "<< fpga_timing.sig_pcie_rd_t << endl;
  cout << endl;

  cout << "- Total API Time: "<< endl;
  cout << "  Runtime             : "<< api_t << endl;
  cout << endl;
}

void disp_results(CONFIG config, cpu_t timing_cpu){

  cout << endl << endl;
  cout << "MEASUREMENTS \n";
  cout << "--------------\n";
  cout << "Points          : " << config.num << "^3\n";
  cout << "Iterations      : " << config.iter << endl << endl;

  cout << "CPU:" << endl;
  cout << "----" << endl;
  cout << "Filter Runtime   : "<< timing_cpu.filter_t << endl;
  cout << "Conv3D Runtime   : "<< timing_cpu.conv_t << endl << endl;
}