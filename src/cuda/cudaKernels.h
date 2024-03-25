#ifndef CUDA_H
#define CUDA_H

#include "core/lbmModel.h"
#include "core/gridGeometry.h"

template<typename T, typename LBMModelClassType>
void launchCreateDeviceModel(LBMModelClassType** deviceModel);

template<typename T, typename LBMModelClassType>
__global__ void createDeviceModel(LBMModelClassType** deviceModel);

template<typename T, typename LBMModelClassType>
void test1(LBMModelClassType* lbmModel);

template<typename T, typename LBMModelClassType>
__global__ void useClass(LBMModelClassType* lbmModel);

template<typename T>
__global__ void initializeLBMDistributions(T* Collide, LBMModel<T>* lbmModel, GridGeometry2D<T>* gridGeometry);

template<typename T,typename LBMModelClassType>
void KERNEL_CALLER_initializeLBMDistributions(T *Collide, LBMModelWrapper<T,LBMModelClassType>* lbmModel, GridGeometry2DWrapper<T>* gridGeometry);

#endif
