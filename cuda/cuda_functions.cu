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
  // upologismoi gia grey
}


__global__ void kernel_rgb(uint8_t* start, uint8_t* end, int height, int width)
{
  // upologismoi gia rgb
}



extern "C" void filtering( uint8_t* table, int height, int width, int loops, char* type )
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

  const int block_size = 16;
  int i;

  for(i=0; i<loops; i++)
  {
    if(strcmp(type, "GREY") == 0)
    {
      int gridX, gridY;
      gridX = (height + block_size - 1) / block_size;
      gridY = (width + block_size - 1) / block_size;
      dim3 dimBlock(block_size, block_size);
      dim3 dimGrid(gridX, gridY);
      kernel_grey<<<dimGrid, dimBlock>>>(start, end, height, width);
    }
    else
    {
      int gridX, gridY;
      gridX = (height + block_size - 1) / block_size;
      gridY = ((width * 3) + block_size - 1) / block_size;
      dim3 dimBlock(block_size, block_size);
      dim3 dimGrid(gridX, gridY);
      kernel_rgb<<<dimGrid, dimBlock>>>(start, end, height, width);
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
