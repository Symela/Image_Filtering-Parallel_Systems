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
  float blur[3][3] = {{1,2,1}, {2,4,2}, {1,2,1}}; // filtro blur gaussian
  float filter[3][3];

  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      filter[i][j]=blur[2-i][2-j];  // antistofi filtrou ana grammes kai ana sthles

  // h thesi tou thread ston pinaka
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  float new_val;
  new_val = 0.0;  // h nea timh sthn thesh tou pinaka

  for (i = x-1, k = 0 ; i <= x+1 ; i++, k++) {
    for (j = y-1, l = 0 ; j <= y+1 ; j++, l++) {
      if ( i == -1 || i == height || j == -1 || j == width ) {
        new_val = start[width * x + y]; // an eisai ektos twn diastasevn tou pinaka,
        //pol/se thn timh tou pixel ths theshs, pou allazeis timh, me thn antistoixi thesi tou filtrou
      }
      else {
        new_val += start[width * i + j] * filter[k][l] / 16.0;  // an eisai entos twn diastasevn tou pinaka,
        //pol/se thn timh tou pixel pou koitas me thn antistoixi thesi tou filtrou
      }
    }
  }
  end[width * x + y] = new_val;
}


__global__ void kernel_rgb(uint8_t* start, uint8_t* end, int height, int width)
{
  int i, j, k, l;
  float blur[3][3] = {{1,2,1}, {2,4,2}, {1,2,1}}; // filtro blur gaussian
  float filter[3][3];

  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      filter[i][j]=blur[2-i][2-j];  // antistofi filtrou ana grammes kai ana sthles

  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  float new_val_red, new_val_blue, new_val_green;
  new_val_red = 0.0;  // h nees times sthn thesh tou pinaka gia red, green and blue
  new_val_blue = 0.0;
  new_val_green = 0.0;

  for (i = x-1, k = 0 ; i <= x+1 ; i++, k++) {
    for (j = (y*3)-3, l = 0 ; j <= (y*3)+3 ; j += 3, l++) {
      if ( i == -1 || i == height || j == -3 || j == (3*width) ) { // an eisai ektos twn diastasevn tou pinaka,
      //pol/se tis times gia red, green and blue tou pixel ths theshs, pou allazeis timh, me thn antistoixi thesi tou filtrou
        new_val_red += start[(width*3) * x + y] * filter[k][l] / 16.0;
        new_val_green += start[(width*3) * x + y + 1] * filter[k][l] / 16.0;
        new_val_blue += start[(width*3) * x + y + 2] * filter[k][l] / 16.0;
      }
      else { // an eisai entos twn diastasevn tou pinaka,
      //pol/se tis times gia red, green and blue tou pixel pou koitas me thn antistoixi thesi tou filtrou
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

  int i;

  for(i=0; i<loops; i++) // gia osa loops zhththikan
  {
    if(strcmp(type, "GREY") == 0) // an einai GREY
    {
      int gridX, gridY;
      gridX = (height + blocksize - 1) / blocksize;
      gridY = (width + blocksize - 1) / blocksize;
      dim3 dimBlock(blocksize, blocksize);
      dim3 dimGrid(gridX, gridY);
      kernel_grey<<<dimGrid, dimBlock>>>(start, end, height, width);
    }
    else  // an einai RGB
    {
      int gridX, gridY;
      gridX = (height + blocksize - 1) / blocksize;
      gridY = ((width * 3) + blocksize - 1) / blocksize;
      dim3 dimBlock(blocksize, blocksize);
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
