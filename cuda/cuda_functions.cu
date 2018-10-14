#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cuda.h>


// __device__ int counter;

__global__ void kernel_grey(uint8_t* start, uint8_t* end, int height, int width)
{
  int i, j, k, l;
  float blur[3][3] = {{1,2,1}, {2,4,2}, {1,2,1}};
  float filter[3][3];

  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      filter[i][j]=blur[2-i][2-j];

  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  float new_val;
  new_val = 0.0;

  for (i = x-1, k = 0 ; i <= x+1 ; i++, k++) {
    for (j = y-1, l = 0 ; j <= y+1 ; j++, l++) {
      if ( i == -1 || i == height || j == -1 || j == width ) {
        new_val = start[width * x + y];
      }
      else {
        new_val += start[width * i + j] * filter[k][l] / 16.0;
      }
    }
  }
  end[width * x + y] = new_val;
  // atomicAdd(&counter, 1);
  // printf("rrr %d\n", counter);
  // // if(new_val == start[width * x + y]) counter++;
}


__global__ void kernel_rgb(uint8_t* start, uint8_t* end, int height, int width)
{
  int i, j, k, l;
  float blur[3][3] = {{1,2,1}, {2,4,2}, {1,2,1}};
  float filter[3][3];

  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      filter[i][j]=blur[2-i][2-j];

  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  float new_val_red, new_val_blue, new_val_green;
  new_val_red = 0.0;
  new_val_blue = 0.0;
  new_val_green = 0.0;

  for (i = x-1, k = 0 ; i <= x+1 ; i++, k++) {
    for (j = (y*3)-3, l = 0 ; j <= (y*3)+3 ; j += 3, l++) {
      if ( i == -1 || i == height || j == -3 || j == (3*width) ) {
        new_val_red += start[(width*3) * x + y] * filter[k][l] / 16.0;
        new_val_green += start[(width*3) * x + y + 1] * filter[k][l] / 16.0;
        new_val_blue += start[(width*3) * x + y + 2] * filter[k][l] / 16.0;
      }
      else {
        new_val_red += start[(width*3) * i + j] * filter[k][l] / 16.0;
        new_val_green += start[(width*3) * i + j + 1] * filter[k][l] / 16.0;
        new_val_blue += start[(width*3) * i + j + 2] * filter[k][l] / 16.0;
      }
    }
  }
  end[(width*3) * x + (y*3)] = new_val_red;
  end[(width*3) * x + (y*3) + 1] = new_val_green;
  end[(width*3) * x + (y*3) + 2] = new_val_blue;
}

//
// __global__ void kernel_grey_equal(uint8_t* start, uint8_t* end, int height, int width, int* flag)
// {
//   size_t x = blockIdx.x * blockDim.x + threadIdx.x;
//   size_t y = blockIdx.y * blockDim.y + threadIdx.y;
//   printf("Hellooo: %d\nS", *flag);
//
//   if(start[width * x + y] == end[width * x + y]) {
//     atomicAdd( flag , 1);
//   }
// }
//
//
// __global__ void kernel_rgb_equal(uint8_t* start, uint8_t* end, int height, int width, int* flag)
// {
//   size_t x = blockIdx.x * blockDim.x + threadIdx.x;
//   size_t y = blockIdx.y * blockDim.y + threadIdx.y;
//
//   if(start[(width*3) * x + y] == end[(width*3) * x + y] && start[(width*3) * x + y + 1] == end[(width*3) * x + y + 1] && start[(width*3) * x + y + 2] == end[(width*3) * x + y + 2] ) atomicAdd( flag , 1);
// }



extern "C" void filtering( uint8_t* table, int height, int width, int loops, char* type, int blocksize)
{
  uint8_t* start, * end, * temp;
  size_t bytes;
  // int flag;

  if(strcmp(type, "GREY") == 0) bytes = height * width;
  else bytes = height * width * 3;

  // desmeuse xoro gia kathe vector sthn GPU
  cudaMalloc( &start, bytes*sizeof(uint8_t) );
  cudaMalloc( &end, bytes*sizeof(uint8_t) );

  // antegrapse ta host vectors sthn mnhmh ths suskeuhs
  cudaMemcpy( start, table, bytes, cudaMemcpyHostToDevice );
  cudaMemset( end, 0, bytes );

  int i;

  for(i=0; i<loops; i++)
  {
    if(strcmp(type, "GREY") == 0)
    {
      int gridX, gridY;
      gridX = (height + blocksize - 1) / blocksize;
      gridY = (width + blocksize - 1) / blocksize;
      dim3 dimBlock(blocksize, blocksize);
      dim3 dimGrid(gridX, gridY);
      // cudaMemset( &counter, 0, sizeof(int) );
      kernel_grey<<<dimGrid, dimBlock>>>(start, end, height, width);
      // cudaMemcpyFromSymbol(&flag, "counter", sizeof(int), 0, cudaMemcpyDeviceToHost );
      // printf("%d\n", flag);
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
    //
    // cudaDeviceSynchronize();
    //
    // if(strcmp(type, "GREY") == 0)
    // {
    //   int gridX, gridY;
    //   gridX = (height + blocksize - 1) / blocksize;
    //   gridY = (width + blocksize - 1) / blocksize;
    //   dim3 dimBlock(blocksize, blocksize);
    //   dim3 dimGrid(gridX, gridY);
    //   printf("YOOOO\n");
    //   kernel_grey_equal<<<dimGrid, dimBlock>>>(start, end, height, width, &flag);
    // }
    // else
    // {
    //   int gridX, gridY;
    //   gridX = (height + blocksize - 1) / blocksize;
    //   gridY = ((width * 3) + blocksize - 1) / blocksize;
    //   dim3 dimBlock(blocksize, blocksize);
    //   dim3 dimGrid(gridX, gridY);
    //   kernel_rgb_equal<<<dimGrid, dimBlock>>>(start, end, height, width, &flag);
    // }

    // antimetathese tous pinakes
    temp = start;
    start = end;
    end = temp;
    //
    // if(flag[0] == (height*width)) break;
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
