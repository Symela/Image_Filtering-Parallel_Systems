#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cuda.h>


__global__ void kernel_grey(uint8_t* start, uint8_t* end, int height, int width)
{
  int i, j, k, l;
  int filter[3][3] = {{1,2,1}, {2,4,2}, {1,2,1}};

  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (0 < x && x < height-1 && 0 < y && y < width-1) {
    float new_val;
    new_val = 0;
    for (i = x-1, k = 0 ; i <= x+1 ; i++, k++) {
      for (j = y-1, l = 0 ; j <= y+1 ; j++, l++) {
        new_val += src[width * i + j] * h[k][l] / 16.0;
      }
    }
    dst[width * x + y] = new_val;
	}
}


__global__ void kernel_rgb(uint8_t* start, uint8_t* end, int height, int width)
{
  int i, j, k, l;
  int filter[3][3] = {{1,2,1}, {2,4,2}, {1,2,1}};

  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (0 < x && x < height-1 && 0 < y && y < width-1) {
    float new_val_red, new_val_blue, new_val_green;
    new_val_red = 0;
    new_val_blue = 0;
    new_val_green = 0;
    for (i = x-1, k = 0 ; i <= x+1 ; i++, k++) {
      for (j = (y*3)-3, l = 0 ; j <= (y*3)+3 ; j += 3, l++) {
        new_val_red += src[(width*3) * i + j] * h[k][l] / 16.0;
        new_val_green += src[(width*3) * i + j + 1] * h[k][l] / 16.0;
        new_val_blue += src[(width*3) * i + j + 2] * h[k][l] / 16.0;
      }
    }
    dst[(width*3) * x + (y*3)] = new_val_red;
    dst[(width*3) * x + (y*3) + 1] = new_val_green;
    dst[(width*3) * x + (y*3) + 2] = new_val_blue;
	}
}



extern "C" void filtering( uint8_t* table, int height, int width, int loops, char* type, int blocksize)
{
  uint8_t* start, * end, * temp;
  size_t bytes;

  if(strcmp(type, "GREY") == 0) bytes = height * width;
  else bytes = height * width * 3;

  // desmeuse xoro gia kathe vector sthn GPU
  cudaMalloc( &start, bytes*sizeof(uint8_t) );
  cudaMalloc( &end, bytes*sizeof(uint8_t) );

  // antegrapse ta host vectors sthn mnhmh ths suskeuhs
  cudaMemcpy( start, table, bytes, cudaMemcpyHostToDevice );
  cudaMemset( end, 0, bytes );

  int i, flag_no_change;
  flag_no_change = 0;

  for(i=0; i<loops && flag_no_change == 0; i++)
  {
    if(strcmp(type, "GREY") == 0)
    {
      int gridX, gridY;
      gridX = (height + blocksize - 1) / blocksize;
      gridY = (width + blocksize - 1) / blocksize;
      dim3 dimBlock(blocksize, blocksize);
      dim3 dimGrid(gridX, gridY);
      kernel_grey<<<dimGrid, dimBlock>>>(start, end, height, width);
    }
    else
    {
      int gridX, gridY;
      gridX = (height + blocksize - 1) / blocksize;
      gridY = ((width * 3) + blocksize - 1) / blocksize;
      dim3 dimBlock(blocksize, blocksize);
      dim3 dimGrid(gridX, gridY);
      kernel_rgb<<<dimGrid, dimBlock>>>(start, end, height, width);
    }

    if( strcmp(start, end) == 0 )
    {
      flag_no_change = 1;
    }
    // antimetathese tous pinakes
    temp = start;
    start = end;
    end = temp;
  }

  cudaGetLastError();
  cudaThreadSynchronize();


  // antegrapse thn telikh eikona ston host
  if(loops%2 == 0)
  {
    cudaMemcpy( table, start, bytes, cudaMemcpyDeviceToHost );
  }
  else
  {
    cudaMemcpy( table, end, bytes, cudaMemcpyDeviceToHost );
  }

  // apeleutherwse thn mnhmh ths suskeuhs
  cudaFree(start);
  cudaFree(end);
}
