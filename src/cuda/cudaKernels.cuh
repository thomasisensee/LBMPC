#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "core/kernelParameters.h"

template<typename T>
__global__ void initializeDistributionsKernel(T* collision, const LBParams<T>* const params);

template<typename T>
void initializeDistributionsCaller(T* deviceCollision, const LBParams<T>* const params, dim3 gridSize, dim3 blockSize);

template<typename T>
__global__ void doStreamingKernel(const T *const collision, T *streaming, const LBParams<T>* const params);

template<typename T>
void doStreamingCaller(T** deviceCollision, T** deviceStreaming, const LBParams<T>* const params, dim3 gridSize, dim3 blockSize);

template<typename T>
__global__ void doCollisionCHMKernel(T* collision, const CollisionParamsCHM<T>* const params);

template<typename T>
__global__ void doCollisionBGKKernel(T* collision, const LBParams<T>* const params);

template<typename T>
void doCollisionBGKCaller(T* deviceCollision, const LBParams<T>* const params, dim3 gridSize, dim3 blockSize);

template<typename T>
void doCollisionCHMCaller(T* deviceCollision, const CollisionParamsCHM<T>* const params, dim3 gridSize, dim3 blockSize);

template<typename T>
__global__ void applyBounceBackKernel(T* collision, const BoundaryParams<T>* const params);

template<typename T>
void applyBounceBackCaller(T* deviceCollision, const BoundaryParams<T>* const params, dim3 gridSize, dim3 blockSize);

template<typename T>
__global__ void computeZerothMomentKernel(T* zerothMoment, const T* const collision, const LBParams<T>* const params);

template<typename T>
void computeZerothMomentCaller(T* deviceZerothMoment, const T* const deviceCollision, const LBParams<T>* const params, dim3 gridSize, dim3 blockSize);

template<typename T>
__global__ void computeFirstMomentKernel(T* firstMoment, const T* const collision, const LBParams<T>* const params);

template<typename T>
void computeFirstMomentCaller(T* deviceFirstMoment, const T* const deviceCollision, const LBParams<T>* const params, dim3 gridSize, dim3 blockSize);

template<typename T>
__global__ void testKernel(T* collision, const LBParams<T>* const params);

template<typename T>
__global__ void testKernel(T* collision, const LBParams<T>* const params);

template<typename T>
void testKernelCaller(T* deviceCollision, const LBParams<T>* const params, dim3 gridSize, dim3 blockSize);

#endif // CUDA_KERNELS_CUH
