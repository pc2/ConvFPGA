#include <iostream>
#include "conv3D.hpp"

int circular(int M, int x){
  if (x<0)
    return x+M;
  if(x >= M)
    return x-M;
  return x;
}

void conv3D(float2 *src, float2 *dst, unsigned N, float *filter, unsigned m){
  float sum, x1, y1;

  for(size_t z = 0; z < N; z++)
    for(size_t y = 0; y < N; y++){
      for(size_t x = 0; x < N; x++){
          sum = 0.0;

        for(int k = -1; k <= 1; k++){
          for(int j = -1; j <= 1; j++){
            for(size_t i = -1; i <= 1; i++){

              /*
              x1 = circular(src.cols, x - j);
              y1 = circular(src.rows, y - k);
              sum = sum + Kernel[j+1][k+1]*src.at<uchar>(y1,x1);
              */

          }
        }
        //dst.at<uchar>(y,x) = sum;

      }
    }
  }
}