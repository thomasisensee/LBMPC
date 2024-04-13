#ifndef PLATFORM_DEFINITIONS_H
#define PLATFORM_DEFINITIONS_H

/// Define preprocessor macros for conditional compilation of device-side functions and constant storage
#ifdef __CUDACC__
  #define any_platform __device__ __host__
  #ifdef __CUDA_ARCH__
    #define platform_constant constexpr __constant__
  #else
    #define platform_constant constexpr
  #endif
#else
  #define any_platform
  #define platform_constant constexpr
#endif

#endif // PLATFORM_DEFINITIONS_H