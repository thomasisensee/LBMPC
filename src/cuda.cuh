#ifndef CUDA_H
#define CUDA_H

#include "cuda_error_handler.h"

template<typename T>
void AllocateDeviceField(T **d_collision,size_t ArraySize)
{
	cudaErrorCheck(cudaMalloc(d_collision, ArraySize));
}

#endif
