#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

//#include "core/lbmModel.h"
//#include "core/gridGeometry.h"
#include "core/simulationParams.h"

template<typename T>
__global__ void initializeDistributionsKernel(T* collision, const FluidParams<T>* const params);

template<typename T>
void initializeDistributionsCaller(T* deviceCollision, const FluidParams<T>* const params, dim3 gridSize, dim3 blockSize);

template<typename T>
__global__ void doStreamingKernel(const T *const collision, T *streaming, const FluidParams<T>* const params);

template<typename T>
void doStreaming(T* deviceCollision, T* deviceStreaming, const FluidParams<T>* const params, dim3 gridSize, dim3 blockSize);


#endif
