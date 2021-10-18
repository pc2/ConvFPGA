//  Author: Arjun Ramaswami
#include <iostream>
#include <iomanip>
#include <math.h>
#include <fftw3.h>
#include "cxxopts.hpp"
#include "helper.hpp"
#include "config.h"

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

/**
 * \brief  parse command line arguments
 * \param  config: configuration struct set on runtime
 */
void parse_args(int argc, char* argv[], CONFIG &config){

  try{
    cxxopts::Options options("Convolution3D", "3D Conv Filter on incoming images");
    options.add_options()
      ("p, path", "Path to bitstream", cxxopts::value<string>())
      ("w, wisdomfile", "File to wisdom", cxxopts::value<string>()->default_value("a.out"))
      ("n, num", "Size of FFT dim", cxxopts::value<unsigned>()->default_value("64"))
      ("i, iter", "Number of iterations", cxxopts::value<unsigned>()->default_value("1"))
      ("t, threads", "Number of threads", cxxopts::value<unsigned>()->default_value("1"))
      ("y, noverify", "No verification", cxxopts::value<bool>()->default_value("false") )
      ("b, batch", "Num of even batches", cxxopts::value<unsigned>()->default_value("1") )
      ("c, cpu-only", "CPU FFTW Only", cxxopts::value<bool>()->default_value("false") )
      ("s, usesvm", "SVM enabled", cxxopts::value<bool>()->default_value("false") )
      ("e, emulate", "toggle emulation", cxxopts::value<bool>()->default_value("false") )
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
    if(opt.count("wisdomfile")){
      config.wisdomfile = opt["wisdomfile"].as<string>();
    }

    config.num = opt["num"].as<unsigned>();
    config.threads = opt["threads"].as<unsigned>();
    config.iter = opt["iter"].as<unsigned>();
    config.batch = opt["batch"].as<unsigned>();
    config.noverify = opt["noverify"].as<bool>();
    config.usesvm = opt["usesvm"].as<bool>();
    config.emulate = opt["emulate"].as<bool>();
  }
  catch(const cxxopts::OptionException& e){
    cerr << "Error parsing options: " << e.what() << endl;
    exit(1);
  }
}

/**
 * \brief  print configuration set by runtime params
 * \param  config: configuration struct set on runtime
 */
void print_config(const CONFIG config){
  cout << endl;
  cout << "CONFIGURATION: \n";
  cout << "---------------\n";
  cout << "Bitstream    = " << config.path << endl;
  cout << "Points       = {"<< config.num << ", " << config.num << ", " << config.num << "}" << endl;
  cout << "Wisdom Path  = " << config.wisdomfile << endl;
  switch(FFTW_PLAN){
    case FFTW_MEASURE:  cout << "FFTW Plan    = Measure\n";
                        break;
    case FFTW_ESTIMATE: cout << "FFTW Plan    = Estimate\n";
                        break;
    case FFTW_PATIENT:  cout << "FFTW Plan    = Patient\n";
                        break;
    case FFTW_EXHAUSTIVE: cout << "FFTW Plan   = Exhaustive\n";
                        break;
    default: throw "-- Incorrect plan set\n";
            break;
  }

  cout << "Threads      = "<< config.threads << endl;
  cout << "Iterations   = "<< config.iter << endl;
  cout << "Batch        = "<< config.batch << endl;
  cout << "----------------\n\n";
}

/**
 * \brief  create random single precision complex floating point values  
 * \param  inp : pointer to float2 data of size N 
 * \param  N   : number of points in the array
 * \param  batch: number of convolutions
 */
void create_data(float2 *inp, const unsigned num_pts, const unsigned batch){

  if(inp == NULL || num_pts < 4 || batch < 1)
    throw "Bad args in create data function";

  for(unsigned j = 0; j < num_pts; j++){
    inp[j].x = (float)((float)rand() / (float)RAND_MAX);
    inp[j].y = (float)((float)rand() / (float)RAND_MAX);
  }


  /*
   * TODO: batched svm doesn't seem to work when using random numbers 
   * over all the batches. Every odd batch has the wrong output in this case.
   * The temporary fix is to use the same signal data in all batches 
   * The error seems to not be cause by some strange data accesses in transpose 
   * buffers between the batches as far as I have experimented. 
   */
  for(unsigned i = 1; i < batch; i++){
    for(unsigned j = 0; j < num_pts; j++){
      inp[(i*num_pts)+j].x = inp[j].x;
      inp[(i*num_pts)+j].y = inp[j].y;
    }
  }
}


/**
 * \brief  display to console the performance measurments from the fpga run
 * \param  config: based on arguments passed at runtime
 * \param  runtime: array of runtime measurements 
 */
void disp_results(const CONFIG config, const fpga_t *runtime){

  fpga_t avg_runtime = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0};

  for(unsigned i = 0; i < config.iter; i++){
    avg_runtime.filter_exec_t += runtime[i].filter_exec_t;
    avg_runtime.filter_pcie_wr_t += runtime[i].filter_pcie_wr_t;

    avg_runtime.sig_exec_t += runtime[i].sig_exec_t;
    avg_runtime.sig_pcie_wr_t += runtime[i].sig_pcie_wr_t;
    avg_runtime.sig_pcie_rd_t += runtime[i].sig_pcie_rd_t;
    avg_runtime.siginv_exec_t += runtime[i].siginv_exec_t;
  } 

  avg_runtime.filter_exec_t = avg_runtime.filter_exec_t / config.iter;
  avg_runtime.filter_pcie_wr_t = avg_runtime.filter_pcie_wr_t / config.iter;
  avg_runtime.sig_exec_t = avg_runtime.sig_exec_t / config.iter;
  avg_runtime.sig_pcie_rd_t = avg_runtime.sig_pcie_rd_t / config.iter;
  avg_runtime.sig_pcie_wr_t = avg_runtime.sig_pcie_wr_t / config.iter;
  avg_runtime.siginv_exec_t = avg_runtime.siginv_exec_t / config.iter;

  if(config.batch > 1 && config.usesvm){
    avg_runtime.siginv_exec_t = 0.0;
    avg_runtime.sig_pcie_rd_t = 0.0;
    avg_runtime.sig_pcie_wr_t = 0.0;
  }

  fpga_t variance = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0};
  fpga_t sd = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0};

  for(unsigned i = 0; i < config.iter; i++){
    variance.filter_exec_t += pow(runtime[i].filter_exec_t - avg_runtime.filter_exec_t, 2);
    variance.filter_pcie_wr_t += pow(runtime[i].filter_pcie_wr_t - avg_runtime.filter_pcie_wr_t, 2);
    variance.sig_exec_t += pow(runtime[i].sig_exec_t - avg_runtime.sig_exec_t, 2);
    variance.sig_pcie_rd_t += pow(runtime[i].sig_pcie_rd_t - avg_runtime.sig_pcie_rd_t, 2);
    variance.sig_pcie_wr_t += pow(runtime[i].sig_pcie_wr_t - avg_runtime.sig_pcie_wr_t, 2);
    variance.siginv_exec_t += pow(runtime[i].siginv_exec_t - avg_runtime.siginv_exec_t, 2);
  }
  sd.filter_exec_t = sqrt(variance.filter_exec_t / config.iter);
  sd.filter_pcie_wr_t = sqrt(variance.filter_pcie_wr_t / config.iter);
  sd.sig_exec_t = sqrt(variance.sig_exec_t / config.iter);
  sd.sig_pcie_rd_t = sqrt(variance.sig_pcie_rd_t / config.iter);
  sd.sig_pcie_wr_t = sqrt(variance.sig_pcie_wr_t / config.iter);
  sd.siginv_exec_t = sqrt(variance.siginv_exec_t / config.iter);

  printf("\n\n------------------------------------------\n");
  printf("Measurements in ms\n");
  printf("--------------------------------------------\n");
  printf("%s", config.iter>1 ? "Average Measurements of iterations\n":"");

  cout << "FPGA:" << endl;
  cout << "-----" << endl;
  cout << "- Filter:" << endl;
  if(!config.usesvm)
    cout << "  PCIe Host to Device : "<< avg_runtime.filter_pcie_wr_t << endl;
  cout << "  Execution           : "<< avg_runtime.filter_exec_t << endl;
  cout << endl;

  cout << "- Signal Convolution:" << endl;
  if(!config.usesvm)
    cout << "  PCIe Host to Device : "<< avg_runtime.sig_pcie_wr_t << endl;
  cout << "  FFT + Conv          : "<< avg_runtime.sig_exec_t << endl;
  cout << "  Inverse FFT         : "<< avg_runtime.siginv_exec_t << endl;
  cout << "  Total Computation   : "<< (avg_runtime.sig_exec_t + avg_runtime.siginv_exec_t) << endl;
  if(config.batch > 1)
    cout << "  Total Comp per batch: "<< (avg_runtime.sig_exec_t + avg_runtime.siginv_exec_t) / config.batch << endl;
  if(!config.usesvm)
    cout << "  PCIe Dev to Host    : "<< avg_runtime.sig_pcie_rd_t << endl;

  if(config.iter > 1){
    printf("\n");
    printf("%s", config.iter>1 ? "Deviation of runtimes among iterations\n":"");
    
    cout << "- Filter:" << endl;
    if(!config.usesvm)
      printf("PCIe Host to Dev    = %.4lfms\n", sd.filter_pcie_wr_t);
    printf("Execution    = %.4lfms\n", sd.filter_exec_t);

    cout << "- Signal:" << endl;
    printf("FFT + Conv          = %.4lfms\n", sd.sig_exec_t);
    printf("Inverse FFT         = %.4lfms\n", sd.siginv_exec_t);
    printf("Total Computation   = %.4lfms\n", (sd.sig_exec_t+sd.siginv_exec_t));

    if(!config.usesvm){
      printf("PCIe Host to Dev      = %.4lfms\n", sd.sig_pcie_wr_t);
      printf("PCIe Dev to Host      = %.4lfms\n", sd.sig_pcie_rd_t);
    }
    cout << endl;
  }

}

/**
 * \brief  display to console the performance measurments from a cpu-only run
 * \param  config: based on arguments passed at runtime
 * \param  timing_cpu: runtime measurement
 */
void disp_results(const CONFIG config, const cpu_t timing_cpu){

  cout << endl << endl;
  cout << "MEASUREMENTS \n";
  cout << "--------------\n";
  cout << "Points           : " << config.num << "^3\n";
  cout << "Threads          : " << config.threads << endl;
  cout << "Iterations       : " << config.iter << endl << endl;

  cout << "CPU:" << endl;
  cout << "----" << endl;
  cout << "Filter Runtime   : "<< timing_cpu.filter_t << "ms" << endl;
  cout << "Conv3D Runtime   : "<< timing_cpu.conv_t << "ms" << endl << endl;
}