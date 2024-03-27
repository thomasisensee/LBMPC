#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

//#include "core/lbmModel.h"
//#include "core/gridGeometry.h"
#include "core/kernelParams.h"

template<typename T>
__global__ void initializeDistributionsKernel(T* collision, const LBMParams<T>* const params);

template<typename T>
void initializeDistributionsCaller(T* deviceCollision, const LBMParams<T>* const params, dim3 gridSize, dim3 blockSize);

template<typename T>
__global__ void doStreamingKernel(const T *const collision, T *streaming, T* swap, const LBMParams<T>* const params);

template<typename T>
void doStreamingCaller(T* deviceCollision, T* deviceStreaming, T* swap, const LBMParams<T>* const params, dim3 gridSize, dim3 blockSize);

template<typename T>
__global__ void doCollisionCHMKernel(T* collision, const CollisionParamsCHM<T>* const params);

template<typename T>
void doCollisionCHMCaller(T* deviceCollision, const CollisionParamsCHM<T>* const params, dim3 gridSize, dim3 blockSize);

#endif
