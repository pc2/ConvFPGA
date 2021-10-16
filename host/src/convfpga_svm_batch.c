// Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#define CL_VERSION_2_0
#include <CL/cl_ext_intelfpga.h> // to disable interleaving & transfer data to specific banks - CL_CHANNEL_1_INTELFPGA
#include "CL/opencl.h"

#include "fpga_state.h"
#include "convfpga/convfpga.h"
#include "svm.h"
#include "opencl_utils.h"
#include "misc.h"

#define WR_GLOBALMEM 0
#define RD_GLOBALMEM 1
#define BATCH 2

#define CHAN_NOT_OUT 0
#define CHAN_OUT 1


/**
 * \brief  compute an out-of-place single precision complex 3D-FFT using the DDR of the FPGA for 3D Transpose
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fpgaf_conv3D_svm_batch(const unsigned N, float2 *sig, float2 *filter, float2 *out, const unsigned how_many) {
  fpga_t conv3D_time = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0};
  cl_int status = 0;
  // if N is not a power of 2
  if(sig == NULL || filter == NULL || out == NULL || ( (N & (N-1)) !=0))
    return conv3D_time;

  // Setup kernels
  cl_kernel fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch1 kernel");
  cl_kernel ffta_kernel = clCreateKernel(program, "fft3da", &status);
  checkError(status, "Failed to create fft3da kernel");
  cl_kernel transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create transpose kernel");
  cl_kernel fftb_kernel = clCreateKernel(program, "fft3db", &status);
  checkError(status, "Failed to create fft3db kernel");
  cl_kernel transpose3D_kernel= clCreateKernel(program, "transpose3D", &status);
  checkError(status, "Failed to create transpose3D kernel");
  cl_kernel fftc_kernel = clCreateKernel(program, "fft3dc", &status);
  checkError(status, "Failed to create fft3dc kernel");
  cl_kernel store_kernel = clCreateKernel(program, "store", &status);
  checkError(status, "Failed to create store kernel");
  cl_kernel conv3D_kernel = clCreateKernel(program, "conv3D", &status);
  checkError(status, "Failed to create conv3D kernel");

  // Setup Queues to the kernels
  queue_setup();

  // Device memory buffers
  const unsigned num_pts = N * N * N;
  cl_mem d_Buf1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  cl_mem d_Buf2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  cl_mem d_Buf3 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_3_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  cl_mem d_Buf4 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_4_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  size_t num_bytes = num_pts * sizeof(float2);
  // allocate SVM buffers
  float2 *filter_inData, *filter_outData;

  filter_inData = (float2 *)clSVMAlloc(context, CL_MEM_READ_ONLY, sizeof(float2) * num_pts, 0);
  filter_outData = (float2 *)clSVMAlloc(context, CL_MEM_WRITE_ONLY, sizeof(float2) * num_pts, 0);

  // Copy Filter Data to SVM Buffer
  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)filter_inData, sizeof(float2) * num_pts, 0, NULL, NULL);
  checkError(status, "Failed to map input data");

  // copy data into h_inData
  memcpy(filter_inData, filter, num_bytes);

  status = clEnqueueSVMUnmap(queue1, (void *)filter_inData, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");

  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)filter_outData, sizeof(float2) * num_pts, 0, NULL, NULL);
  checkError(status, "Failed to map input data");

  // copy data into h_inData
  memset(&filter_outData[0], 0, num_bytes);

  status = clEnqueueSVMUnmap(queue1, (void *)filter_outData, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");

  // Step 2: Transform filter: Buf1 -> Buf2 -> Buf1
  int inverse_int_a = 0;
  int inverse_int_b = 0;
  int use_svm_a = 1;
  status = clSetKernelArgSVMPointer(fetch_kernel, 0, (void *)filter_inData);
  checkError(status, "Failed to set fetch kernel arg");
  status=clSetKernelArg(fetch_kernel, 1, sizeof(cl_mem), (void *)&d_Buf1);
  checkError(status, "Failed to set fetch kernel arg 1");
  status=clSetKernelArg(fetch_kernel, 2, sizeof(cl_int), (void*)&use_svm_a);
  checkError(status, "Failed to set transpose3D kernel arg 2");

  status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int_a);
  checkError(status, "Failed to set ffta kernel arg");
  status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int_a);
  checkError(status, "Failed to set fftb kernel arg");

  // - Writing to Buf2 from Buf1 before transpose
  int mode = WR_GLOBALMEM;
  status=clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf3);
  checkError(status, "Failed to set transpose3D kernel arg 0");
  status=clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), (void *)&d_Buf2);
  checkError(status, "Failed to set transpose3D kernel arg 0");
  status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode);
  checkError(status, "Failed to set transpose3D kernel arg 2");

  status=clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int_a);
  checkError(status, "Failed to set fftc kernel arg");

  int chan_out = CHAN_NOT_OUT;
  int use_svm_b = 0;
  status = clSetKernelArgSVMPointer(store_kernel, 0, (void *)filter_outData);
  checkError(status, "Failed to set store kernel arg 0");
  status=clSetKernelArg(store_kernel, 1, sizeof(cl_mem), (void *)&d_Buf1);
  checkError(status, "Failed to set store kernel arg 1");
  status=clSetKernelArg(store_kernel, 2, sizeof(cl_int), (void *)&chan_out);
  checkError(status, "Failed to set store kernel arg 2");
  status=clSetKernelArg(store_kernel, 3, sizeof(cl_int), (void *)&use_svm_b);
  checkError(status, "Failed to set store kernel arg 3");

  // Kernel Execution
  cl_event startExec_event, endExec_event;

  status = clEnqueueTask(queue7, store_kernel, 0, NULL, &endExec_event);
  checkError(status, "Failed to launch transpose kernel");

  status = clEnqueueTask(queue6, fftc_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  // Write: Bank 1 to 2
  status = clEnqueueTask(queue5, transpose3D_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second transpose kernel");

  // Read from 2 to 3
  mode = RD_GLOBALMEM;
  status=clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf2);
  checkError(status, "Failed to set transpose3D kernel arg 0");
  status=clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), (void *)&d_Buf3);
  checkError(status, "Failed to set transpose3D kernel arg 0");
  status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode);
  checkError(status, "Failed to set transpose3D kernel arg 2");

  status = clEnqueueTask(queue5, transpose3D_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose3D kernel");

  status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second fft kernel");

  status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, &startExec_event);
  checkError(status, "Failed to launch fetch kernel");

  status = clFinish(queue1);
  checkError(status, "failed to finish");
  status = clFinish(queue2);
  checkError(status, "failed to finish");
  status = clFinish(queue3);
  checkError(status, "failed to finish");
  status = clFinish(queue4);
  checkError(status, "failed to finish");
  status = clFinish(queue5);
  checkError(status, "failed to finish");
  status = clFinish(queue6);
  checkError(status, "failed to finish");
  status = clFinish(queue7);
  checkError(status, "failed to finish");

  cl_ulong kernel_start = 0, kernel_end = 0;

  clGetEventProfilingInfo(startExec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
  clGetEventProfilingInfo(endExec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);
  
  conv3D_time.filter_exec_t = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06);

  if (filter_inData)
    clSVMFree(context, filter_inData);
  if (filter_outData)
    clSVMFree(context, filter_outData);

  // Step 3: Transform Signal and stream to convolution kernel
  //                             Buf1 (Filter)
  //                              |
  //  Buf1 -> Buf2 -> chanout -> .* -> Buf4 

  // Copy Signal to SVM Buffer

  num_bytes = num_pts * sizeof(float2);

  float2 *h_inData[how_many], *h_outData[how_many];
  for(size_t i = 0; i < how_many; i++){
    
    h_inData[i] = (float2 *)clSVMAlloc(context, CL_MEM_READ_ONLY, sizeof(float2) * num_pts, 0);
    h_outData[i] = (float2 *)clSVMAlloc(context, CL_MEM_WRITE_ONLY, sizeof(float2) * num_pts, 0);

    status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_inData[i], sizeof(float2) * num_pts, 0, NULL, NULL);
    checkError(status, "Failed to map input data");

    // copy data into h_inData
    memcpy(&h_inData[i][0], &sig[i*num_pts], num_bytes);

    status = clEnqueueSVMUnmap(queue1, (void *)h_inData[i], 0, NULL, NULL);
    checkError(status, "Failed to unmap input data");

    status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_outData[i], sizeof(float2) * num_pts, 0, NULL, NULL);
    checkError(status, "Failed to map input data");

    // set h_outData to 0
    memset(&h_outData[i][0], 0, num_bytes);

    status = clEnqueueSVMUnmap(queue1, (void *)h_outData[i], 0, NULL, NULL);
    checkError(status, "Failed to unmap input data");
  }

  /*
   * First phase of FFT3D i.e. until Transpose3D
   */
  use_svm_a = 1;
  inverse_int_a = 0;
  status = clSetKernelArgSVMPointer(fetch_kernel, 0, (void *)h_inData[0]);
  checkError(status, "Failed to set fetch kernel arg");
  status=clSetKernelArg(fetch_kernel, 1, sizeof(cl_mem), (void *)&d_Buf1);
  checkError(status, "Failed to set fetch kernel arg 1");
  status=clSetKernelArg(fetch_kernel, 2, sizeof(cl_int), (void*)&use_svm_a);
  checkError(status, "Failed to set transpose3D kernel arg 2");

  status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int_a);
  checkError(status, "Failed to set ffta kernel arg");
  status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int_a);
  checkError(status, "Failed to set fftb kernel arg");

  // - Writing to Buf2 from SVM Host before transpose
  mode = WR_GLOBALMEM;
  status=clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf4);
  checkError(status, "Failed to set transpose3D kernel arg 0");
  status=clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), (void *)&d_Buf2);
  checkError(status, "Failed to set transpose3D kernel arg 0");
  status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode);
  checkError(status, "Failed to set transpose3D kernel arg 2");

  // Enqueue Tasks
  status = clEnqueueTask(queue5, transpose3D_kernel, 0, NULL, &endExec_event);
  checkError(status, "Failed to launch transpose3D write kernel");

  status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second fft kernel");

  status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, &startExec_event);
  checkError(status, "Failed to launch fetch kernel");

  status = clFinish(queue1);
  checkError(status, "failed to finish");
  status = clFinish(queue2);
  checkError(status, "failed to finish");
  status = clFinish(queue3);
  checkError(status, "failed to finish");
  status = clFinish(queue4);
  checkError(status, "failed to finish");
  status = clFinish(queue5);
  checkError(status, "failed to finish");

  clGetEventProfilingInfo(startExec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
  clGetEventProfilingInfo(endExec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);
  
  double first_ever = (cl_double)(kernel_start);
  double last_ever = 0.0;
  double temp = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06);
  double total_temp = temp;

  for(size_t i = 1; i < how_many; i+=2){
    printf("Batch: %lu\n", i);
    /*
    * Overlap FFT+conv with the first phase of the second batch of input
    */
    use_svm_a = 1;
    inverse_int_a = 0;
    status = clSetKernelArgSVMPointer(fetch_kernel, 0, (void *)h_inData[i]);
    checkError(status, "Failed to set fetch kernel arg");
    status=clSetKernelArg(fetch_kernel, 1, sizeof(cl_mem), (void *)&d_Buf1);
    checkError(status, "Failed to set fetch kernel arg 1");
    status=clSetKernelArg(fetch_kernel, 2, sizeof(cl_int), (void*)&use_svm_a);
    checkError(status, "Failed to set transpose3D kernel arg 2");

    status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int_a);
    checkError(status, "Failed to set ffta kernel arg");
    status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int_a);
    checkError(status, "Failed to set fftb kernel arg");

    mode = BATCH;
    status=clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf2);
    checkError(status, "Failed to set transpose3D kernel arg 0");
    status=clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), (void *)&d_Buf4);
    checkError(status, "Failed to set transpose3D kernel arg 0");
    status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode);
    checkError(status, "Failed to set transpose3D kernel arg 2");

    inverse_int_b = 0;
    chan_out = CHAN_OUT;
    use_svm_b = 0;
    status=clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int_b);
    checkError(status, "Failed to set fftc kernel arg");

    status = clSetKernelArgSVMPointer(store_kernel, 0, (void *)h_outData[i]);
    checkError(status, "Failed to set store kernel arg 0");
    status=clSetKernelArg(store_kernel, 1, sizeof(cl_mem), (void *)&d_Buf3);
    checkError(status, "Failed to set store kernel arg 1");
    status=clSetKernelArg(store_kernel, 2, sizeof(cl_int), (void *)&chan_out);
    checkError(status, "Failed to set store kernel arg 2");
    status=clSetKernelArg(store_kernel, 3, sizeof(cl_int), (void *)&use_svm_b);
    checkError(status, "Failed to set store kernel arg 3");

    status=clSetKernelArg(conv3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf1);
    checkError(status, "Failed to set conv3D kernel arg 0");

    status=clSetKernelArg(conv3D_kernel, 1, sizeof(cl_mem), (void *)&d_Buf3);
    checkError(status, "Failed to set conv3D kernel arg 0");

    // Enqueue
    status = clEnqueueTask(queue8, conv3D_kernel, 0, NULL, &endExec_event);
    checkError(status, "Failed to launch conv3D kernel");

    status = clEnqueueTask(queue7, store_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch store kernel");

    status = clEnqueueTask(queue6, fftc_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fft kernel");

    status = clEnqueueTask(queue5, transpose3D_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch transpose3D write kernel");

    status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch second fft kernel");

    status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch transpose kernel");

    status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fft kernel");

    status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, &startExec_event);
    checkError(status, "Failed to launch fetch kernel");

    status = clFinish(queue1);
    checkError(status, "failed to finish");
    status = clFinish(queue2);
    checkError(status, "failed to finish");
    status = clFinish(queue3);
    checkError(status, "failed to finish");
    status = clFinish(queue4);
    checkError(status, "failed to finish");
    status = clFinish(queue5);
    checkError(status, "failed to finish");
    status = clFinish(queue6);
    checkError(status, "failed to finish");
    status = clFinish(queue7);
    checkError(status, "failed to finish");
    status = clFinish(queue8);
    checkError(status, "failed to finish");

    clGetEventProfilingInfo(startExec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
    clGetEventProfilingInfo(endExec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);
    
    temp = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06);
    total_temp += temp;

    // Step 3: Inv FFT3D first phase: second phase of second batch
    use_svm_a = 0;
    inverse_int_a = 1;
    status = clSetKernelArgSVMPointer(fetch_kernel, 0, (void *)h_inData[i]);
    checkError(status, "Failed to set fetch kernel arg");
    status=clSetKernelArg(fetch_kernel, 1, sizeof(cl_mem), (void *)&d_Buf3);
    checkError(status, "Failed to set fetch kernel arg 1");
    status=clSetKernelArg(fetch_kernel, 2, sizeof(cl_int), (void*)&use_svm_a);
    checkError(status, "Failed to set transpose3D kernel arg 2");

    status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int_a);
    checkError(status, "Failed to set ffta kernel arg");
    status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int_a);
    checkError(status, "Failed to set fftb kernel arg");

    mode = BATCH;
    status=clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf4);
    checkError(status, "Failed to set transpose3D kernel arg 0");
    status=clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), (void *)&d_Buf2);
    checkError(status, "Failed to set transpose3D kernel arg 0");
    status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode);
    checkError(status, "Failed to set transpose3D kernel arg 2");

    inverse_int_b = 0;
    chan_out = CHAN_OUT;
    use_svm_b = 0;
    status=clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int_b);
    checkError(status, "Failed to set fftc kernel arg");
    status = clSetKernelArgSVMPointer(store_kernel, 0, (void*)h_outData[i]);
    checkError(status, "Failed to set store kernel arg");
    status=clSetKernelArg(store_kernel, 1, sizeof(cl_mem), (void *)&d_Buf2);
    checkError(status, "Failed to set store kernel arg");
    status=clSetKernelArg(store_kernel, 2, sizeof(cl_int), (void *)&chan_out);
    checkError(status, "Failed to set store kernel arg 1");
    status=clSetKernelArg(store_kernel, 3, sizeof(cl_int), (void *)&use_svm_b);
    checkError(status, "Failed to set store kernel arg 3");
    
    status=clSetKernelArg(conv3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf1);
    checkError(status, "Failed to set conv3D kernel arg 0");

    status=clSetKernelArg(conv3D_kernel, 1, sizeof(cl_mem), (void *)&d_Buf3);
    checkError(status, "Failed to set conv3D kernel arg 0");

    // Kernel Execution
    status = clEnqueueTask(queue8, conv3D_kernel, 0, NULL, &endExec_event);
    checkError(status, "Failed to launch conv3D_kernel kernel");

    status = clEnqueueTask(queue7, store_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch transpose kernel");

    status = clEnqueueTask(queue6, fftc_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fft kernel");

    status = clEnqueueTask(queue5, transpose3D_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch transpose3D kernel");

    status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch second fft kernel");

    status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch transpose kernel");

    status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fft kernel");

    status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, &startExec_event);
    checkError(status, "Failed to launch fetch kernel");

    status = clFinish(queue1);
    checkError(status, "failed to finish");
    status = clFinish(queue2);
    checkError(status, "failed to finish");
    status = clFinish(queue3);
    checkError(status, "failed to finish");
    status = clFinish(queue4);
    checkError(status, "failed to finish");
    status = clFinish(queue5);
    checkError(status, "failed to finish");
    status = clFinish(queue6);
    checkError(status, "failed to finish");
    status = clFinish(queue7);
    checkError(status, "failed to finish");
    status = clFinish(queue8);
    checkError(status, "failed to finish");

    clGetEventProfilingInfo(startExec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
    clGetEventProfilingInfo(endExec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);
    
    temp = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06);
    total_temp += temp;

    /*
    *  First batch final and second batch ifft
    */
    use_svm_a = 0;
    inverse_int_a = 1;
    status = clSetKernelArgSVMPointer(fetch_kernel, 0, (void *)h_inData[i]);
    checkError(status, "Failed to set fetch kernel arg");
    status=clSetKernelArg(fetch_kernel, 1, sizeof(cl_mem), (void *)&d_Buf3);
    checkError(status, "Failed to set fetch kernel arg 1");
    status=clSetKernelArg(fetch_kernel, 2, sizeof(cl_int), (void*)&use_svm_a);
    checkError(status, "Failed to set transpose3D kernel arg 2");

    status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int_a);
    checkError(status, "Failed to set ffta kernel arg");
    status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int_a);
    checkError(status, "Failed to set fftb kernel arg");

    mode = BATCH;
    status=clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf2);
    checkError(status, "Failed to set transpose3D kernel arg 0");
    status=clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), (void *)&d_Buf4);
    checkError(status, "Failed to set transpose3D kernel arg 0");
    status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode);
    checkError(status, "Failed to set transpose3D kernel arg 2");

    inverse_int_b = 1;
    chan_out = CHAN_NOT_OUT;
    use_svm_b = 1;
    status=clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int_b);
    checkError(status, "Failed to set fftc kernel arg");

    // kernel stores using SVM based PCIe to host
    status = clSetKernelArgSVMPointer(store_kernel, 0, (void*)h_outData[i-1]);
    checkError(status, "Failed to set store kernel arg");
    status=clSetKernelArg(store_kernel, 1, sizeof(cl_mem), (void *)&d_Buf2);
    checkError(status, "Failed to set store kernel arg");
    status=clSetKernelArg(store_kernel, 2, sizeof(cl_int), (void *)&chan_out);
    checkError(status, "Failed to set store kernel arg 1");
    status=clSetKernelArg(store_kernel, 3, sizeof(cl_int), (void *)&use_svm_b);
    checkError(status, "Failed to set store kernel arg 3");

    status = clEnqueueTask(queue7, store_kernel, 0, NULL, &endExec_event);
    checkError(status, "Failed to launch transpose kernel");

    status = clEnqueueTask(queue6, fftc_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fft kernel");

    status = clEnqueueTask(queue5, transpose3D_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch transpose3D kernel");

    status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch second fft kernel");

    status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch transpose kernel");

    status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fft kernel");

    status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, &startExec_event);
    checkError(status, "Failed to launch fetch kernel");

    status = clFinish(queue1);
    checkError(status, "failed to finish");
    status = clFinish(queue2);
    checkError(status, "failed to finish");
    status = clFinish(queue3);
    checkError(status, "failed to finish");
    status = clFinish(queue4);
    checkError(status, "failed to finish");
    status = clFinish(queue5);
    checkError(status, "failed to finish");
    status = clFinish(queue6);
    checkError(status, "failed to finish");
    status = clFinish(queue7);
    checkError(status, "failed to finish");

    clGetEventProfilingInfo(startExec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
    clGetEventProfilingInfo(endExec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);
    
    temp = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06);
    total_temp += temp;

    /* Final Output with start of next batch
    */
    chan_out = CHAN_NOT_OUT;
    use_svm_a = 1;
    inverse_int_a = 0;

    // - Writing to Buf2 from SVM Host before transpose
    if(i == (how_many - 1)){
      mode = RD_GLOBALMEM;
    }
    else{
      status = clSetKernelArgSVMPointer(fetch_kernel, 0, (void *)h_inData[i+1]);
      checkError(status, "Failed to set fetch kernel arg");
      status=clSetKernelArg(fetch_kernel, 1, sizeof(cl_mem), (void *)&d_Buf1);
      checkError(status, "Failed to set fetch kernel arg 1");
      status=clSetKernelArg(fetch_kernel, 2, sizeof(cl_int), (void*)&use_svm_a);
      checkError(status, "Failed to set transpose3D kernel arg 2");

      status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int_a);
      checkError(status, "Failed to set ffta kernel arg");
      status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int_a);
      checkError(status, "Failed to set fftb kernel arg");
      mode = BATCH;
    }
    status=clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf4);
    checkError(status, "Failed to set transpose3D kernel arg 0");
    status=clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), (void *)&d_Buf2);
    checkError(status, "Failed to set transpose3D kernel arg 0");
    status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode);
    checkError(status, "Failed to set transpose3D kernel arg 2");

    use_svm_b = 1;
    inverse_int_b = 1;
    status=clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int_b);
    checkError(status, "Failed to set fftc kernel arg");
    // kernel stores using SVM based PCIe to host
    status = clSetKernelArgSVMPointer(store_kernel, 0, (void*)h_outData[i]);
    checkError(status, "Failed to set store kernel arg");
    status=clSetKernelArg(store_kernel, 1, sizeof(cl_mem), (void *)&d_Buf2);
    checkError(status, "Failed to set store kernel arg");
    status=clSetKernelArg(store_kernel, 2, sizeof(cl_int), (void *)&chan_out);
    checkError(status, "Failed to set store kernel arg 1");
    status=clSetKernelArg(store_kernel, 3, sizeof(cl_int), (void *)&use_svm_b);
    checkError(status, "Failed to set store kernel arg 3");

    // Kernel Execution
    status = clEnqueueTask(queue7, store_kernel, 0, NULL, &endExec_event);
    checkError(status, "Failed to launch transpose kernel");

    status = clEnqueueTask(queue6, fftc_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fft kernel");

    status = clEnqueueTask(queue5, transpose3D_kernel, 0, NULL, &startExec_event);
    checkError(status, "Failed to launch transpose3D kernel");

    if(i != (how_many - 1)){
      status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
      checkError(status, "Failed to launch second fft kernel");

      status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
      checkError(status, "Failed to launch transpose kernel");

      status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
      checkError(status, "Failed to launch fft kernel");

      status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, &startExec_event);
      checkError(status, "Failed to launch fetch kernel");

      status = clFinish(queue1);
      checkError(status, "failed to finish");
      status = clFinish(queue2);
      checkError(status, "failed to finish");
      status = clFinish(queue3);
      checkError(status, "failed to finish");
      status = clFinish(queue4);
      checkError(status, "failed to finish");
    }

    status = clFinish(queue5);
    checkError(status, "failed to finish");
    status = clFinish(queue6);
    checkError(status, "failed to finish");
    status = clFinish(queue7);
    checkError(status, "failed to finish");

    clGetEventProfilingInfo(startExec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
    clGetEventProfilingInfo(endExec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);
    temp = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06);
    last_ever = (cl_double)(kernel_end);
    total_temp += temp;  
  }
 
  conv3D_time.sig_exec_t = total_temp;
  conv3D_time.siginv_exec_t = (cl_double)(last_ever - first_ever) * (cl_double)(1e-06) ;
  
  for(size_t i = 0; i < how_many; i++){

    // copy data into h_outData
    status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_READ,
      (void *)h_outData[i], sizeof(float2) * num_pts, 0, NULL, NULL);
    checkError(status, "Failed to map out data");

    memcpy(&out[i*num_pts], &h_outData[i][0], num_bytes);

    status = clEnqueueSVMUnmap(queue1, (void *)h_outData[i], 0, NULL, NULL);
    checkError(status, "Failed to unmap out data");
  }

  for(size_t i = 0; i < how_many; i++){
    clSVMFree(context, h_inData[i]);
    clSVMFree(context, h_outData[i]);
  }

  /*
  printf("Output\n");
  for(unsigned i = 0; i < how_many * num_pts; i++){
    printf("%u: (%f, %f)\n", i, out[i].x, out[i].y);
  }
  printf("\n");
  */
  queue_cleanup();

  if (d_Buf1)
    clReleaseMemObject(d_Buf1);
  if (d_Buf2) 
    clReleaseMemObject(d_Buf2);
  if (d_Buf3) 
    clReleaseMemObject(d_Buf3);
  if (d_Buf4) 
    clReleaseMemObject(d_Buf4);

  if(fetch_kernel) 
    clReleaseKernel(fetch_kernel);  
  if(ffta_kernel) 
    clReleaseKernel(ffta_kernel);  
  if(fftb_kernel) 
    clReleaseKernel(fftb_kernel);  
  if(fftc_kernel) 
    clReleaseKernel(fftc_kernel);  

  if(transpose_kernel) 
    clReleaseKernel(transpose_kernel);  
  if(transpose3D_kernel) 
    clReleaseKernel(transpose3D_kernel);  

  if(store_kernel) 
    clReleaseKernel(store_kernel);  
  if (conv3D_kernel) 
    clReleaseKernel(conv3D_kernel);

  conv3D_time.valid = 1;
  return conv3D_time;
}