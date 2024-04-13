#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "core/kernelParameters.h"

/************************************/
/***** Initialize Distributions *****/
/************************************/
template<typename T,typename LATTICE_DESCRIPTOR>
__global__ void initializeDistributionsKernel(T* collision, const CollisionParamsBGK<T>* const params);

template<typename T,typename LATTICE_DESCRIPTOR>
void initializeDistributionsCaller(T* deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize);

/*********************/
/***** Streaming *****/
/*********************/
template<typename T,typename LATTICE_DESCRIPTOR>
__global__ void doStreamingKernel(const T *const collision, T *streaming, const CollisionParamsBGK<T>* const params);

template<typename T,typename LATTICE_DESCRIPTOR>
void doStreamingCaller(T** deviceCollision, T** deviceStreaming, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize);

/**************************/
/***** Collision: BGK *****/
/**************************/
template<typename T,typename LATTICE_DESCRIPTOR>
__global__ void doCollisionBGKKernel(T* collision, const CollisionParamsBGK<T>* const params);

template<typename T,typename LATTICE_DESCRIPTOR>
void doCollisionBGKCaller(T* deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize);

/**************************/
/***** Collision: CHM *****/
/**************************/
template<typename T,typename LATTICE_DESCRIPTOR>
__global__ void doCollisionCHMKernel(T* collision, const CollisionParamsCHM<T>* const params);

template<typename T,typename LATTICE_DESCRIPTOR>
void doCollisionCHMCaller(T* deviceCollision, const CollisionParamsCHM<T>* const params, dim3 gridSize, dim3 blockSize);


/*******************************/
/***** Boundary conditions *****/
/*******************************/
template<typename T,typename LATTICE_DESCRIPTOR>
__device__ void applyBC(T* collision, const BoundaryParams<T>* const params, unsigned int i, unsigned int j);

template<typename T,typename LATTICE_DESCRIPTOR>
__global__ void applyBounceBackKernel(T* collision, const BoundaryParams<T>* const params);

template<typename T,typename LATTICE_DESCRIPTOR>
void applyBounceBackCaller(T* deviceCollision, const BoundaryParams<T>* const params, dim3 gridSize, dim3 blockSize);

/***************************************/
/***** Moment computations: zeroth *****/
/***************************************/
template<typename T,typename LATTICE_DESCRIPTOR>
__global__ void computeZerothMomentKernel(T* zerothMoment, const T* const collision, const CollisionParamsBGK<T>* const params);

template<typename T,typename LATTICE_DESCRIPTOR>
void computeZerothMomentCaller(T* deviceZerothMoment, const T* const deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize);

/**************************************/
/***** Moment computations: First *****/
/**************************************/
template<typename T,typename LATTICE_DESCRIPTOR>
__global__ void computeFirstMomentKernel(T* firstMoment, const T* const collision, const CollisionParamsBGK<T>* const params);

template<typename T,typename LATTICE_DESCRIPTOR>
void computeFirstMomentCaller(T* deviceFirstMoment, const T* const deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize);

#endif // CUDA_KERNELS_CUH