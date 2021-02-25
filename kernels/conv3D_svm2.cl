// Author: Arjun Ramaswami

#include "conv_config.h"
#include "fft/fft_8.cl" 
#include "matrixTranspose/diagonal_bitrev.cl"

#pragma OPENCL EXTENSION cl_intel_channels : enable

channel float2 chaninfft3da[POINTS]; 
channel float2 chaninfft3db[POINTS];
channel float2 chaninfft3dc[POINTS];

channel float2 chaninTranspose[POINTS];
channel float2 chaninTranspose3D[POINTS];
channel float2 chaninStore[POINTS];

channel float2 chaninConv[POINTS];

#define WR_GLOBALMEM 0
#define RD_GLOBALMEM 1
#define BATCH 2

// Kernel that fetches data from global memory 
kernel void fetch(
  __global __attribute__((buffer_location(SVM_HOST_BUFFER_LOCATION))) volatile float2 * restrict src_host,
  __global __attribute__((buffer_location(DDR_BUFFER_LOCATION))) volatile float2 * restrict src_ddr,
  const int use_svm) {
  unsigned delay = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bitrevA = false;

  float2 __attribute__((memory, numbanks(8))) buf[2][N];
  
  // additional iterations to fill the buffers
  for(unsigned step = 0; step < (N * DEPTH) + delay; step++){

    unsigned where = (step & ((N * DEPTH) - 1)) * 8; 

    float2x8 data;
    
    if (step < (N * DEPTH) && use_svm == 0) {
      data.i0 = src_ddr[where + 0];
      data.i1 = src_ddr[where + 1];
      data.i2 = src_ddr[where + 2];
      data.i3 = src_ddr[where + 3];
      data.i4 = src_ddr[where + 4];
      data.i5 = src_ddr[where + 5];
      data.i6 = src_ddr[where + 6];
      data.i7 = src_ddr[where + 7];
    } 
    if(step < (N * DEPTH) && use_svm == 1){
      data.i0 = src_host[where + 0];
      data.i1 = src_host[where + 1];
      data.i2 = src_host[where + 2];
      data.i3 = src_host[where + 3];
      data.i4 = src_host[where + 4];
      data.i5 = src_host[where + 5];
      data.i6 = src_host[where + 6];
      data.i7 = src_host[where + 7];
    } 
    if(step >= (N * DEPTH)) {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }

    /*
    if (step < (N * DEPTH) && use_svm == 0) {
      data.i0 = src_ddr[where + 0];
      data.i1 = src_ddr[where + 1];
      data.i2 = src_ddr[where + 2];
      data.i3 = src_ddr[where + 3];
      data.i4 = src_ddr[where + 4];
      data.i5 = src_ddr[where + 5];
      data.i6 = src_ddr[where + 6];
      data.i7 = src_ddr[where + 7];
    } 
    else if(step < (N * DEPTH) && use_svm == 1){
      data.i0 = src_host[where + 0];
      data.i1 = src_host[where + 1];
      data.i2 = src_host[where + 2];
      data.i3 = src_host[where + 3];
      data.i4 = src_host[where + 4];
      data.i5 = src_host[where + 5];
      data.i6 = src_host[where + 6];
      data.i7 = src_host[where + 7];
    } 
    else{
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }
    */

    is_bitrevA = ( (step & ((N / 8) - 1)) == 0) ? !is_bitrevA: is_bitrevA;

    unsigned row = step & (DEPTH - 1);
    data = bitreverse_fetch(data,
      is_bitrevA ? buf[0] : buf[1], 
      is_bitrevA ? buf[1] : buf[0], 
      row);

    if (step >= delay) {
      write_channel_intel(chaninfft3da[0], data.i0);
      write_channel_intel(chaninfft3da[1], data.i1);
      write_channel_intel(chaninfft3da[2], data.i2);
      write_channel_intel(chaninfft3da[3], data.i3);
      write_channel_intel(chaninfft3da[4], data.i4);
      write_channel_intel(chaninfft3da[5], data.i5);
      write_channel_intel(chaninfft3da[6], data.i6);
      write_channel_intel(chaninfft3da[7], data.i7);
    }
  }
}

kernel void fft3da(int inverse) {

  /* The FFT engine requires a sliding window for data reordering; data stored
   * in this array is carried across loop iterations and shifted by 1 element
   * every iteration; all loop dependencies derived from the uses of this 
   * array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];

  #pragma loop_coalesce
  for(unsigned j = 0; j < N; j++){
    for (unsigned i = 0; i < N * (N / POINTS) + N / POINTS - 1; i++) {
      float2x8 data;

      if (i < N * (N / POINTS)) {
        data.i0 = read_channel_intel(chaninfft3da[0]);
        data.i1 = read_channel_intel(chaninfft3da[1]);
        data.i2 = read_channel_intel(chaninfft3da[2]);
        data.i3 = read_channel_intel(chaninfft3da[3]);
        data.i4 = read_channel_intel(chaninfft3da[4]);
        data.i5 = read_channel_intel(chaninfft3da[5]);
        data.i6 = read_channel_intel(chaninfft3da[6]);
        data.i7 = read_channel_intel(chaninfft3da[7]);
      } 
      else {
        data.i0 = data.i1 = data.i2 = data.i3 = 
                  data.i4 = data.i5 = data.i6 = data.i7 = 0;
      }

      data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

      // Write result to channels
      if (i >= N / POINTS - 1) {
        write_channel_intel(chaninTranspose[0], data.i0);
        write_channel_intel(chaninTranspose[1], data.i1);
        write_channel_intel(chaninTranspose[2], data.i2);
        write_channel_intel(chaninTranspose[3], data.i3);
        write_channel_intel(chaninTranspose[4], data.i4);
        write_channel_intel(chaninTranspose[5], data.i5);
        write_channel_intel(chaninTranspose[6], data.i6);
        write_channel_intel(chaninTranspose[7], data.i7);
      }
    }
  }
}

kernel void transpose() {
  const int DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bufA = false, is_bitrevA = false;

  float2 buf[2][DEPTH][POINTS];
  //float2 bitrev_in[2][N], bitrev_out[2][N];
  float2 __attribute__((memory, numbanks(8))) bitrev_in[2][N];
  //float2 bitrev_in[2][N];
  float2 __attribute__((memory, numbanks(8))) bitrev_out[2][N];
  
  int initial_delay = DELAY + DELAY; // for each of the bitrev buffer

  // additional iterations to fill the buffers
  for(int step = -initial_delay; step < ((N * DEPTH) + DEPTH); step++){

    float2x8 data, data_out;
    if (step < ((N * DEPTH) - initial_delay)) {
      data.i0 = read_channel_intel(chaninTranspose[0]);
      data.i1 = read_channel_intel(chaninTranspose[1]);
      data.i2 = read_channel_intel(chaninTranspose[2]);
      data.i3 = read_channel_intel(chaninTranspose[3]);
      data.i4 = read_channel_intel(chaninTranspose[4]);
      data.i5 = read_channel_intel(chaninTranspose[5]);
      data.i6 = read_channel_intel(chaninTranspose[6]);
      data.i7 = read_channel_intel(chaninTranspose[7]);
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }

    // Swap buffers every N*N/8 iterations 
    // starting from the additional delay of N/8 iterations
    is_bufA = (( (step + DELAY) & (DEPTH - 1)) == 0) ? !is_bufA: is_bufA;

    // Swap bitrev buffers every N/8 iterations
    is_bitrevA = ( (step & ((N / 8) - 1)) == 0) ? !is_bitrevA: is_bitrevA;

    unsigned row = step & (DEPTH - 1);
    data = bitreverse_in(data,
      is_bitrevA ? bitrev_in[0] : bitrev_in[1], 
      is_bitrevA ? bitrev_in[1] : bitrev_in[0], 
      row);

    writeBuf(data,
      is_bufA ? buf[0] : buf[1],
      step, DELAY);

    data_out = readBuf(
      is_bufA ? buf[1] : buf[0], 
      step);

    unsigned start_row = (step + DELAY) & (DEPTH -1);
    data_out = bitreverse_out(
      is_bitrevA ? bitrev_out[0] : bitrev_out[1],
      is_bitrevA ? bitrev_out[1] : bitrev_out[0],
      data_out, start_row);


    if (step >= (DEPTH)) {
      write_channel_intel(chaninfft3db[0], data_out.i0);
      write_channel_intel(chaninfft3db[1], data_out.i1);
      write_channel_intel(chaninfft3db[2], data_out.i2);
      write_channel_intel(chaninfft3db[3], data_out.i3);
      write_channel_intel(chaninfft3db[4], data_out.i4);
      write_channel_intel(chaninfft3db[5], data_out.i5);
      write_channel_intel(chaninfft3db[6], data_out.i6);
      write_channel_intel(chaninfft3db[7], data_out.i7);
    }
  }
}

kernel void fft3db(int inverse) {

  /* The FFT engine requires a sliding window for data reordering; data stored
   * in this array is carried across loop iterations and shifted by 1 element
   * every iteration; all loop dependencies derived from the uses of this 
   * array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];

  #pragma loop_coalesce
  for(unsigned j = 0; j < N; j++){
    for (unsigned i = 0; i < N * (N / POINTS) + N / POINTS - 1; i++) {
      float2x8 data;

      if (i < N * (N / POINTS)) {
        data.i0 = read_channel_intel(chaninfft3db[0]);
        data.i1 = read_channel_intel(chaninfft3db[1]);
        data.i2 = read_channel_intel(chaninfft3db[2]);
        data.i3 = read_channel_intel(chaninfft3db[3]);
        data.i4 = read_channel_intel(chaninfft3db[4]);
        data.i5 = read_channel_intel(chaninfft3db[5]);
        data.i6 = read_channel_intel(chaninfft3db[6]);
        data.i7 = read_channel_intel(chaninfft3db[7]);
      } else {
        data.i0 = data.i1 = data.i2 = data.i3 = 
                  data.i4 = data.i5 = data.i6 = data.i7 = 0;
      }

      data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

      if (i >= N / POINTS - 1) {
        write_channel_intel(chaninTranspose3D[0], data.i0);
        write_channel_intel(chaninTranspose3D[1], data.i1);
        write_channel_intel(chaninTranspose3D[2], data.i2);
        write_channel_intel(chaninTranspose3D[3], data.i3);
        write_channel_intel(chaninTranspose3D[4], data.i4);
        write_channel_intel(chaninTranspose3D[5], data.i5);
        write_channel_intel(chaninTranspose3D[6], data.i6);
        write_channel_intel(chaninTranspose3D[7], data.i7);
      }
    }
  }
}

kernel void transpose3D(
  __global __attribute__((buffer_location(DDR_BUFFER_LOCATION))) float2 * restrict src, 
  __global __attribute__((buffer_location(DDR_BUFFER_LOCATION))) float2 * restrict dest, const int mode){

  //int N = (1 << LOGN);
  //int DEPTH = (1 << (LOGN + LOGN - LOGPOINTS));
  const int initial_delay = (1 << (LOGN - LOGPOINTS)); // N / 8 for the bitrev buffers
  bool is_bufA = false, is_bitrevA = false;
  bool is_bufB = false, is_bitrevB = false;

  float2 buf_wr[2][DEPTH][POINTS];
  float2 buf_rd[2][DEPTH][POINTS];

  float2 __attribute__((memory, numbanks(8))) bitrev_in[2][N];
  //float2 bitrev_in[2][N];
  float2 __attribute__((memory, numbanks(8))) bitrev_out[2][N];

  const int EXIT_COND = ((N * DEPTH) + DEPTH);

  // additional iterations to fill the buffers
  for(int step = -initial_delay; step < EXIT_COND; step++){

    float2x8 data, data_out;
    float2x8 data_wr, data_wr_out;
    if(mode == WR_GLOBALMEM || mode == BATCH){
      if (step < ((N * DEPTH) - initial_delay)) {
        data.i0 = read_channel_intel(chaninTranspose3D[0]);
        data.i1 = read_channel_intel(chaninTranspose3D[1]);
        data.i2 = read_channel_intel(chaninTranspose3D[2]);
        data.i3 = read_channel_intel(chaninTranspose3D[3]);
        data.i4 = read_channel_intel(chaninTranspose3D[4]);
        data.i5 = read_channel_intel(chaninTranspose3D[5]);
        data.i6 = read_channel_intel(chaninTranspose3D[6]);
        data.i7 = read_channel_intel(chaninTranspose3D[7]);
      } else {
        data.i0 = data.i1 = data.i2 = data.i3 = 
                  data.i4 = data.i5 = data.i6 = data.i7 = 0;
      }

      // Swap buffers every N*N/8 iterations 
      // starting from the additional delay of N/8 iterations
      is_bufA = (( step & (DEPTH - 1)) == 0) ? !is_bufA: is_bufA;

      // Swap bitrev buffers every N/8 iterations
      is_bitrevA = ( (step & ((N / 8) - 1)) == 0) ? !is_bitrevA: is_bitrevA;

      unsigned row = step & (DEPTH - 1);
      data = bitreverse_in(data,
        is_bitrevA ? bitrev_in[0] : bitrev_in[1], 
        is_bitrevA ? bitrev_in[1] : bitrev_in[0], 
        row);

      writeBuf(data,
        is_bufA ? buf_wr[0] : buf_wr[1],
        step, 0);

      data_out = readBuf_store(
        is_bufA ? buf_wr[1] : buf_wr[0], 
        step);

      if (step >= (DEPTH)) {
        unsigned index = (step - DEPTH) * 8;

        dest[index + 0] = data_out.i0;
        dest[index + 1] = data_out.i1;
        dest[index + 2] = data_out.i2;
        dest[index + 3] = data_out.i3;
        dest[index + 4] = data_out.i4;
        dest[index + 5] = data_out.i5;
        dest[index + 6] = data_out.i6;
        dest[index + 7] = data_out.i7;
      }
    } // condition for writing to global memory
    if(mode == RD_GLOBALMEM || mode == BATCH){

      unsigned step_rd = step + initial_delay;
      // increment z by 1 every N/8 steps until (N*N/ 8)
      unsigned start_index = step_rd + initial_delay;
      unsigned zdim = (step_rd >> (LOGN - LOGPOINTS)) & (N - 1); 

      // increment y by 1 every N*N/8 points until N
      unsigned ydim = (step_rd >> (LOGN + LOGN - LOGPOINTS)) & (N - 1);

      // increment by 8 until N / 8
      unsigned xdim = (step_rd * 8) & (N - 1);

      // increment by 1 every N*N*N / 8 steps
      unsigned batch_index = (step_rd >> (LOGN + LOGN + LOGN - LOGPOINTS));

      unsigned index_wr = (batch_index * N * N * N) + (zdim * N * N) + (ydim * N) + xdim; 

      //float2x8 data, data_out;
      if (step < (N * DEPTH)) {
        data_wr.i0 = src[index_wr + 0];
        data_wr.i1 = src[index_wr + 1];
        data_wr.i2 = src[index_wr + 2];
        data_wr.i3 = src[index_wr + 3];
        data_wr.i4 = src[index_wr + 4];
        data_wr.i5 = src[index_wr + 5];
        data_wr.i6 = src[index_wr + 6];
        data_wr.i7 = src[index_wr + 7];
      } else {
        data_wr.i0 = data_wr.i1 = data_wr.i2 = data_wr.i3 = 
                  data_wr.i4 = data_wr.i5 = data_wr.i6 = data_wr.i7 = 0;
      }
    
      is_bufB = (( step_rd & (DEPTH - 1)) == 0) ? !is_bufB: is_bufB;

      // Swap bitrev buffers every N/8 iterations
      is_bitrevB = ( (step_rd & ((N / 8) - 1)) == 0) ? !is_bitrevB: is_bitrevB;

      writeBuf(data_wr,
        is_bufB ? buf_rd[0] : buf_rd[1],
        step_rd, 0);

      data_wr_out = readBuf_fetch(
        is_bufB ? buf_rd[1] : buf_rd[0], 
        step_rd, 0);

      unsigned start_row = step_rd & (DEPTH -1);
      data_wr_out = bitreverse_out(
        is_bitrevB ? bitrev_out[0] : bitrev_out[1],
        is_bitrevB ? bitrev_out[1] : bitrev_out[0],
        data_wr_out, start_row);

      if (step_rd >= (DEPTH + initial_delay)) {
        write_channel_intel(chaninfft3dc[0], data_wr_out.i0);
        write_channel_intel(chaninfft3dc[1], data_wr_out.i1);
        write_channel_intel(chaninfft3dc[2], data_wr_out.i2);
        write_channel_intel(chaninfft3dc[3], data_wr_out.i3);
        write_channel_intel(chaninfft3dc[4], data_wr_out.i4);
        write_channel_intel(chaninfft3dc[5], data_wr_out.i5);
        write_channel_intel(chaninfft3dc[6], data_wr_out.i6);
        write_channel_intel(chaninfft3dc[7], data_wr_out.i7);
      }

    } // condition for reading from global memory
  }
}

kernel void fft3dc(int inverse) {

  /* The FFT engine requires a sliding window for data reordering; data stored
   * in this array is carried across loop iterations and shifted by 1 element
   * every iteration; all loop dependencies derived from the uses of this 
   * array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];

  #pragma loop_coalesce
  for(unsigned j = 0; j < N; j++){

    for (unsigned i = 0; i < N * (N / POINTS) + N / POINTS - 1; i++) {
      float2x8 data;

      if (i < N * (N / POINTS)) {
        data.i0 = read_channel_intel(chaninfft3dc[0]);
        data.i1 = read_channel_intel(chaninfft3dc[1]);
        data.i2 = read_channel_intel(chaninfft3dc[2]);
        data.i3 = read_channel_intel(chaninfft3dc[3]);
        data.i4 = read_channel_intel(chaninfft3dc[4]);
        data.i5 = read_channel_intel(chaninfft3dc[5]);
        data.i6 = read_channel_intel(chaninfft3dc[6]);
        data.i7 = read_channel_intel(chaninfft3dc[7]);
      } else {
        data.i0 = data.i1 = data.i2 = data.i3 = 
                  data.i4 = data.i5 = data.i6 = data.i7 = 0;
      }

      // Perform one FFT step
      data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

      // Write result to channels
      if (i >= N / POINTS - 1) {
        write_channel_intel(chaninStore[0], data.i0);
        write_channel_intel(chaninStore[1], data.i1);
        write_channel_intel(chaninStore[2], data.i2);
        write_channel_intel(chaninStore[3], data.i3);
        write_channel_intel(chaninStore[4], data.i4);
        write_channel_intel(chaninStore[5], data.i5);
        write_channel_intel(chaninStore[6], data.i6);
        write_channel_intel(chaninStore[7], data.i7);
      }
    }
  }
}

kernel void store(
  __global __attribute__((buffer_location(SVM_HOST_BUFFER_LOCATION))) volatile float2 * restrict dest_host,
  __global __attribute__((buffer_location(DDR_BUFFER_LOCATION))) volatile float2 * restrict dest_ddr, const int channel_out, const int use_svm) {

  const int DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bufA = false, is_bitrevA = false;

  float2 buf[2][DEPTH][POINTS];
  //float2 bitrev_in[2][N];
  float2 __attribute__((memory, numbanks(8))) bitrev_in[2][N];
  
  int initial_delay = DELAY; // for each of the bitrev buffer
  // additional iterations to fill the buffers
  for(int step = -initial_delay; step < ((N * DEPTH) + DEPTH); step++){

    float2x8 data, data_out;
    if (step < ((N * DEPTH) - initial_delay)) {
      data.i0 = read_channel_intel(chaninStore[0]);
      data.i1 = read_channel_intel(chaninStore[1]);
      data.i2 = read_channel_intel(chaninStore[2]);
      data.i3 = read_channel_intel(chaninStore[3]);
      data.i4 = read_channel_intel(chaninStore[4]);
      data.i5 = read_channel_intel(chaninStore[5]);
      data.i6 = read_channel_intel(chaninStore[6]);
      data.i7 = read_channel_intel(chaninStore[7]);
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }
    // Swap buffers every N*N/8 iterations 
    // starting from the additional delay of N/8 iterations
    is_bufA = (( step & (DEPTH - 1)) == 0) ? !is_bufA: is_bufA;

    // Swap bitrev buffers every N/8 iterations
    is_bitrevA = ( (step & ((N / 8) - 1)) == 0) ? !is_bitrevA: is_bitrevA;

    unsigned row = step & (DEPTH - 1);
    data = bitreverse_in(data,
      is_bitrevA ? bitrev_in[0] : bitrev_in[1], 
      is_bitrevA ? bitrev_in[1] : bitrev_in[0], 
      row);

    writeBuf(data,
      is_bufA ? buf[0] : buf[1],
      step, 0);

    data_out = readBuf_store(
      is_bufA ? buf[1] : buf[0], 
      step);

    if (step >= (DEPTH)) {
      unsigned start_index = (step - DEPTH);
      // increment z by 1 every N/8 steps until (N*N/ 8)
      unsigned zdim = (start_index >> (LOGN - LOGPOINTS)) & (N - 1); 

      // increment y by 1 every N*N/8 points until N
      unsigned ydim = (start_index >> (LOGN + LOGN - LOGPOINTS)) & (N - 1);

      // incremenet by 8 until N / 8
      unsigned xdim = (start_index * 8) & ( N - 1);
      //unsigned index = (step - DEPTH) * 8;

      // increment by N*N*N
      unsigned cube = LOGN + LOGN + LOGN - LOGPOINTS;

      // increment by 1 every N*N*N / 8 steps
      unsigned batch_index = (start_index >> cube);
      //unsigned batch_index = 0;

      unsigned index = (batch_index * N * N * N) + (zdim * N * N) + (ydim * N) + xdim; 

      if(!channel_out && !use_svm){
        dest_ddr[index + 0] = data_out.i0;
        dest_ddr[index + 1] = data_out.i1;
        dest_ddr[index + 2] = data_out.i2;
        dest_ddr[index + 3] = data_out.i3;
        dest_ddr[index + 4] = data_out.i4;
        dest_ddr[index + 5] = data_out.i5;
        dest_ddr[index + 6] = data_out.i6;
        dest_ddr[index + 7] = data_out.i7;
      }
      if(!channel_out && use_svm){
        dest_host[index + 0] = data_out.i0;
        dest_host[index + 1] = data_out.i1;
        dest_host[index + 2] = data_out.i2;
        dest_host[index + 3] = data_out.i3;
        dest_host[index + 4] = data_out.i4;
        dest_host[index + 5] = data_out.i5;
        dest_host[index + 6] = data_out.i6;
        dest_host[index + 7] = data_out.i7;
      }
      if(channel_out){
        write_channel_intel(chaninConv[0], data_out.i0);
        write_channel_intel(chaninConv[1], data_out.i1);
        write_channel_intel(chaninConv[2], data_out.i2);
        write_channel_intel(chaninConv[3], data_out.i3);
        write_channel_intel(chaninConv[4], data_out.i4);
        write_channel_intel(chaninConv[5], data_out.i5);
        write_channel_intel(chaninConv[6], data_out.i6);
        write_channel_intel(chaninConv[7], data_out.i7);
      }
    }
  }
}

kernel void conv3D(
  __global __attribute__((buffer_location(DDR_BUFFER_LOCATION))) volatile float2 * restrict src,
  __global __attribute__((buffer_location(DDR_BUFFER_LOCATION))) volatile float2 * restrict dest){

  float2x8 filter, signal;
  const int DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8

  for(unsigned i = 0; i < (N * DEPTH); i++){

    unsigned step_rd = i;
    // increment z by 1 every N/8 steps until (N*N/ 8)
    unsigned zdim = (step_rd >> (LOGN - LOGPOINTS)) & (N - 1); 

    // increment y by 1 every N*N/8 points until N
    unsigned ydim = (step_rd >> (LOGN + LOGN - LOGPOINTS)) & (N - 1);

    // increment by 8 until N / 8
    unsigned xdim = (step_rd * 8) & (N - 1);

    // increment by 1 every N*N*N / 8 steps
    unsigned batch_index = (step_rd >> (LOGN + LOGN + LOGN - LOGPOINTS));

    unsigned index_rd = (batch_index * N * N * N) + (zdim * N * N) + (ydim * N) + xdim; 
    //unsigned index_rd = i;

    filter.i0 = src[index_rd + 0];
    filter.i1 = src[index_rd + 1];
    filter.i2 = src[index_rd + 2];
    filter.i3 = src[index_rd + 3];
    filter.i4 = src[index_rd + 4];
    filter.i5 = src[index_rd + 5];
    filter.i6 = src[index_rd + 6];
    filter.i7 = src[index_rd + 7];

    signal.i0 = read_channel_intel(chaninConv[0]);
    signal.i1 = read_channel_intel(chaninConv[1]);
    signal.i2 = read_channel_intel(chaninConv[2]);
    signal.i3 = read_channel_intel(chaninConv[3]);
    signal.i4 = read_channel_intel(chaninConv[4]);
    signal.i5 = read_channel_intel(chaninConv[5]);
    signal.i6 = read_channel_intel(chaninConv[6]);
    signal.i7 = read_channel_intel(chaninConv[7]);

    // Perform complex multiplication
    dest[index_rd + 0] = comp_mult(signal.i0, filter.i0);
    dest[index_rd + 1] = comp_mult(signal.i1, filter.i1);
    dest[index_rd + 2] = comp_mult(signal.i2, filter.i2);
    dest[index_rd + 3] = comp_mult(signal.i3, filter.i3);
    dest[index_rd + 4] = comp_mult(signal.i4, filter.i4);
    dest[index_rd + 5] = comp_mult(signal.i5, filter.i5);
    dest[index_rd + 6] = comp_mult(signal.i6, filter.i6);
    dest[index_rd + 7] = comp_mult(signal.i7, filter.i7);
  }
}

