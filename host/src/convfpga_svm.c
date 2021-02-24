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
fpga_t fpgaf_conv3D_svm(unsigned N, float2 *sig, float2 *filter, float2 *out) {
  fpga_t conv3D_time = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0};
  cl_int status = 0;
  // if N is not a power of 2
  if(sig == NULL || filter == NULL || out == NULL || ( (N & (N-1)) !=0)){
    return conv3D_time;
  }

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
  unsigned num_pts = N * N * N;
  cl_mem d_Buf1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  cl_mem d_Buf2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  cl_mem d_Buf3 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_3_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  cl_mem d_Buf4 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_4_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  num_pts = N * N * N;
  size_t num_bytes = num_pts * sizeof(float2);
  // allocate SVM buffers
  float2 *h_inData, *h_outData;
  h_inData = (float2 *)clSVMAlloc(context, CL_MEM_READ_ONLY, sizeof(float2) * num_pts, 0);
  h_outData = (float2 *)clSVMAlloc(context, CL_MEM_WRITE_ONLY, sizeof(float2) * num_pts, 0);

  num_bytes = num_pts * sizeof(float2);

  // Copy Filter Data to SVM Buffer
  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_inData, sizeof(float2) * num_pts, 0, NULL, NULL);
  checkError(status, "Failed to map input data");

  // copy data into h_inData
  memcpy(h_inData, filter, num_bytes);

  status = clEnqueueSVMUnmap(queue1, (void *)h_inData, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");

  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_outData, sizeof(float2) * num_pts, 0, NULL, NULL);
  checkError(status, "Failed to map input data");

  // copy data into h_inData
  memset(&h_outData[0], 0, num_bytes);

  status = clEnqueueSVMUnmap(queue1, (void *)h_outData, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");

  // Step 2: Transform filter: Buf1 -> Buf2 -> Buf3
  int inverse_int = 0;
  int use_svm = 1;
  status = clSetKernelArgSVMPointer(fetch_kernel, 0, (void *)h_inData);
  checkError(status, "Failed to set fetch kernel arg");
  status=clSetKernelArg(fetch_kernel, 1, sizeof(cl_mem), (void *)&d_Buf1);
  checkError(status, "Failed to set fetch kernel arg 1");
  status=clSetKernelArg(fetch_kernel, 2, sizeof(cl_int), (void*)&use_svm);
  checkError(status, "Failed to set transpose3D kernel arg 2");

  status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set ffta kernel arg");
  status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftb kernel arg");

  // - Writing to Buf2 from Buf1 before transpose
  int mode = WR_GLOBALMEM;
  status=clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf3);
  checkError(status, "Failed to set transpose3D kernel arg 0");
  status=clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), (void *)&d_Buf2);
  checkError(status, "Failed to set transpose3D kernel arg 0");
  status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode);
  checkError(status, "Failed to set transpose3D kernel arg 2");

  status=clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftc kernel arg");

  int chan_out = CHAN_NOT_OUT;
  use_svm = 0;
  status = clSetKernelArgSVMPointer(store_kernel, 0, (void *)h_outData);
  checkError(status, "Failed to set store kernel arg 0");
  status=clSetKernelArg(store_kernel, 1, sizeof(cl_mem), (void *)&d_Buf3);
  checkError(status, "Failed to set store kernel arg 1");
  status=clSetKernelArg(store_kernel, 2, sizeof(cl_int), (void *)&chan_out);
  checkError(status, "Failed to set store kernel arg 2");
  status=clSetKernelArg(store_kernel, 3, sizeof(cl_int), (void *)&use_svm);
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

  // Step 3: Transform Signal and stream to convolution kernel
  //                             Buf3 (Filter)
  //                              |
  //  Buf1 -> Buf2 -> chanout -> .* -> Buf4 

  // Copy Signal to SVM Buffer
  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_inData, sizeof(float2) * num_pts, 0, NULL, NULL);
  checkError(status, "Failed to map input data");

  // copy data into h_inData
  memcpy(h_inData, sig, num_bytes);

  status = clEnqueueSVMUnmap(queue1, (void *)h_inData, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");

  use_svm = 1;
  status = clSetKernelArgSVMPointer(fetch_kernel, 0, (void *)h_inData);
  checkError(status, "Failed to set fetch kernel arg");
  status=clSetKernelArg(fetch_kernel, 1, sizeof(cl_mem), (void *)&d_Buf1);
  checkError(status, "Failed to set fetch kernel arg 1");
  status=clSetKernelArg(fetch_kernel, 2, sizeof(cl_int), (void*)&use_svm);
  checkError(status, "Failed to set transpose3D kernel arg 2");

  status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set ffta kernel arg");
  status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftb kernel arg");

  // - Writing to Buf2 from SVM Host before transpose
  mode = WR_GLOBALMEM;
  status=clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf4);
  checkError(status, "Failed to set transpose3D kernel arg 0");
  status=clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), (void *)&d_Buf2);
  checkError(status, "Failed to set transpose3D kernel arg 0");
  status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode);
  checkError(status, "Failed to set transpose3D kernel arg 2");

  status=clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftc kernel arg");

  chan_out = CHAN_OUT;
  use_svm = 0;
  status = clSetKernelArgSVMPointer(store_kernel, 0, (void *)h_outData);
  checkError(status, "Failed to set store kernel arg 0");
  status=clSetKernelArg(store_kernel, 1, sizeof(cl_mem), (void *)&d_Buf4);
  checkError(status, "Failed to set store kernel arg 1");
  status=clSetKernelArg(store_kernel, 2, sizeof(cl_int), (void *)&chan_out);
  checkError(status, "Failed to set store kernel arg 2");
  status=clSetKernelArg(store_kernel, 3, sizeof(cl_int), (void *)&use_svm);
  checkError(status, "Failed to set store kernel arg 3");

  // Filter Buf3
  status=clSetKernelArg(conv3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf3);
  checkError(status, "Failed to set conv3D kernel arg 0");

  // Output in Buf4
  status=clSetKernelArg(conv3D_kernel, 1, sizeof(cl_mem), (void *)&d_Buf4);
  checkError(status, "Failed to set conv3D kernel arg 0");
  // Read from 1 + chan out -> write to 4

  status = clEnqueueTask(queue8, conv3D_kernel, 0, NULL, &endExec_event);
  checkError(status, "Failed to launch conv3D kernel");

  status = clEnqueueTask(queue7, store_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch store kernel");

  status = clEnqueueTask(queue6, fftc_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue5, transpose3D_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose3D write kernel");

  mode = RD_GLOBALMEM;
  status=clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf2);
  checkError(status, "Failed to set transpose3D kernel arg 0");
  status=clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), (void *)&d_Buf1);
  checkError(status, "Failed to set transpose3D kernel arg 0");
  status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode);
  checkError(status, "Failed to set transpose3D kernel arg 2");

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
  
  conv3D_time.sig_exec_t = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06);

  // Step 4: Inverse FFT
  // Buf4 -> Buf1 -> Host
  // Output in buffer 2
  use_svm = 0;
  status = clSetKernelArgSVMPointer(fetch_kernel, 0, (void *)h_inData);
  checkError(status, "Failed to set fetch kernel arg");
  status=clSetKernelArg(fetch_kernel, 1, sizeof(cl_mem), (void *)&d_Buf4);
  checkError(status, "Failed to set fetch kernel arg 1");
  status=clSetKernelArg(fetch_kernel, 2, sizeof(cl_int), (void*)&use_svm);
  checkError(status, "Failed to set transpose3D kernel arg 2");

  inverse_int = 1;
  status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set ffta kernel arg");
  status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftb kernel arg");

  // - Writing to Buf4 to Buf1
  mode = WR_GLOBALMEM;
  status=clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf3);
  checkError(status, "Failed to set transpose3D kernel arg 0");
  status=clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), (void *)&d_Buf1);
  checkError(status, "Failed to set transpose3D kernel arg 0");
  status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode);
  checkError(status, "Failed to set transpose3D kernel arg 2");

  status=clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftc kernel arg");

  // - Output to Buf2
  chan_out = CHAN_NOT_OUT;
  use_svm = 1;
  // kernel stores using SVM based PCIe to host
  status = clSetKernelArgSVMPointer(store_kernel, 0, (void*)h_outData);
  checkError(status, "Failed to set store kernel arg");
  status=clSetKernelArg(store_kernel, 1, sizeof(cl_mem), (void *)&d_Buf2);
  checkError(status, "Failed to set store kernel arg");
  status=clSetKernelArg(store_kernel, 2, sizeof(cl_int), (void *)&chan_out);
  checkError(status, "Failed to set store kernel arg 1");
  status=clSetKernelArg(store_kernel, 3, sizeof(cl_int), (void *)&use_svm);
  checkError(status, "Failed to set store kernel arg 3");

  // Kernel Execution
  status = clEnqueueTask(queue7, store_kernel, 0, NULL, &endExec_event);
  checkError(status, "Failed to launch transpose kernel");

  status = clEnqueueTask(queue6, fftc_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue5, transpose3D_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose3D kernel");

  mode = RD_GLOBALMEM;
  status=clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), (void *)&d_Buf1);
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

  clGetEventProfilingInfo(startExec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
  clGetEventProfilingInfo(endExec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);
  
  conv3D_time.siginv_exec_t = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06);

  // Copy results from device to host
  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_READ,
    (void *)h_outData, sizeof(float2) * num_pts, 0, NULL, NULL);
  checkError(status, "Failed to map out data");

  memcpy(out, h_outData, num_bytes);

  status = clEnqueueSVMUnmap(queue1, (void *)h_outData, 0, NULL, NULL);
  checkError(status, "Failed to unmap out data");

  if (h_inData)
    clSVMFree(context, h_inData);
  if (h_outData)
    clSVMFree(context, h_outData);

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