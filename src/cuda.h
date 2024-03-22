#ifndef CUDA_H
#define CUDA_H

template<typename T>
void AllocateDeviceField(T** d_array, size_t ArraySize);

template<typename T>
void FreeDeviceField(T* d_array);

#endif
