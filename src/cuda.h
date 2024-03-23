#ifndef CUDA_H
#define CUDA_H

#include "lbmModel.h"
#include "gridGeometry.h"

template<typename T>
void allocateDeviceField(T** d_array, size_t ArraySize);

template<typename T>
void freeDeviceField(T* d_array);

template<typename T>
void KERNEL_CALLER_initializeLBMDistributions(T *Collide, LBMModel<T>* lbmModel, GridGeometry2D<T>* gridGeometry);

#endif
