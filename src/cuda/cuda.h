#ifndef CUDA_H
#define CUDA_H

#include "core/lbmModel.h"
#include "core/gridGeometry.h"

template<typename T>
void KERNEL_CALLER_initializeLBMDistributions(T *Collide, LBMModelWrapper<T>* lbmModel, GridGeometry2DWrapper<T>* gridGeometry);

#endif
