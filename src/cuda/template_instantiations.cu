#include "cudaKernels.cuh"
//#include "cudaKernels.cu"
#include "core/descriptors/aliases.h"

template void initializeDistributionsCaller<float,descriptors::D2Q9Standard<float>>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void initializeDistributionsCaller<float,descriptors::D2Q5Standard<float>>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void initializeDistributionsCaller<double,descriptors::D2Q9Standard<double>>(double* deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);
template void initializeDistributionsCaller<double,descriptors::D2Q5Standard<double>>(double* deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);

template void doStreamingCaller<float,descriptors::D2Q9Standard<float>>(float** deviceCollision, float** deviceStreaming, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void doStreamingCaller<float,descriptors::D2Q5Standard<float>>(float** deviceCollision, float** deviceStreaming, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void doStreamingCaller<double,descriptors::D2Q9Standard<double>>(double** deviceCollision, double** deviceStreaming, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);
template void doStreamingCaller<double,descriptors::D2Q5Standard<double>>(double** deviceCollision, double** deviceStreaming, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);

template void doCollisionBGKCaller<float,descriptors::D2Q9Standard<float>>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionBGKCaller<float,descriptors::D2Q5Standard<float>>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionBGKCaller<double,descriptors::D2Q9Standard<double>>(double* deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionBGKCaller<double,descriptors::D2Q5Standard<double>>(double* deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);

template void doCollisionCHMCaller<float,descriptors::D2Q9Standard<float>>(float* deviceCollision, const CollisionParamsCHM<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionCHMCaller<float,descriptors::D2Q5Standard<float>>(float* deviceCollision, const CollisionParamsCHM<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionCHMCaller<double,descriptors::D2Q9Standard<double>>(double* deviceCollision, const CollisionParamsCHM<double>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionCHMCaller<double,descriptors::D2Q5Standard<double>>(double* deviceCollision, const CollisionParamsCHM<double>* const params, dim3 gridSize, dim3 blockSize);

template void applyBounceBackCaller<float,descriptors::D2Q9Standard<float>>(float* deviceCollision, const BoundaryParams<float>* const params, dim3 gridSize, dim3 blockSize);
template void applyBounceBackCaller<float,descriptors::D2Q5Standard<float>>(float* deviceCollision, const BoundaryParams<float>* const params, dim3 gridSize, dim3 blockSize);
template void applyBounceBackCaller<double,descriptors::D2Q9Standard<double>>(double* deviceCollision, const BoundaryParams<double>* const params, dim3 gridSize, dim3 blockSize);
template void applyBounceBackCaller<double,descriptors::D2Q5Standard<double>>(double* deviceCollision, const BoundaryParams<double>* const params, dim3 gridSize, dim3 blockSize);

template void computeZerothMomentCaller<float,descriptors::D2Q9Standard<float>>(float* deviceZerothMoment, const float* const deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void computeZerothMomentCaller<float,descriptors::D2Q5Standard<float>>(float* deviceZerothMoment, const float* const deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void computeZerothMomentCaller<double,descriptors::D2Q9Standard<double>>(double* deviceZerothMoment, const double* const deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);
template void computeZerothMomentCaller<double,descriptors::D2Q5Standard<double>>(double* deviceZerothMoment, const double* const deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);

template void computeFirstMomentCaller<float,descriptors::D2Q9Standard<float>>(float* deviceFirstMoment, const float* const deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void computeFirstMomentCaller<float,descriptors::D2Q5Standard<float>>(float* deviceFirstMoment, const float* const deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void computeFirstMomentCaller<double,descriptors::D2Q9Standard<double>>(double* deviceFirstMoment, const double* const deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);
template void computeFirstMomentCaller<double,descriptors::D2Q5Standard<double>>(double* deviceFirstMoment, const double* const deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);