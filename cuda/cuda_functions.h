#ifndef __CUDA_FUNCTIONS__
#define __CUDA_FUNCTIONS__

#include <cuda.h>
#include "cuda_runtime.h"

  extern "C" void filtering( uint8_t* table, int height, int width, int loops, char* type );

#endif
