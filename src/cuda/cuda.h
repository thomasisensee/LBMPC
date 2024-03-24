#ifndef CUDA_H
#define CUDA_H

#include "core/lbmModel.h"
#include "core/gridGeometry.h"

template<typename T>
void allocateDeviceField(T** d_array, size_t ArraySize);

template<typename T>
void freeDeviceField(T* d_array);

template<typename T>
void KERNEL_CALLER_initializeLBMDistributions(T *Collide, LBMModelWrapper<T>* lbmModel, GridGeometry2DWrapper<T>* gridGeometry);

#endif
