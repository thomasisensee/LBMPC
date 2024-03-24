#ifndef CUDA_H
#define CUDA_H

#include "core/lbmModel.h"
#include "core/gridGeometry.h"

template<typename T>
void KERNEL_CALLER_test(LBMModel<T>* lbmModel);

template<typename T>
__global__ void useClass(LBMModel<T>* lbmModel);

template<typename T>
__global__ void testKernel(T a);

template<typename T>
__global__ void initializeLBMDistributions(T* Collide, LBMModel<T>* lbmModel, GridGeometry2D<T>* gridGeometry);

template<typename T>
void KERNEL_CALLER_initializeLBMDistributions(T *Collide, LBMModelWrapper<T>* lbmModel, GridGeometry2DWrapper<T>* gridGeometry);

#endif
