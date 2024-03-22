#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda.h"

/**
 * Checks the returned cudaError_t and prints corresponding message in case of error.
 */
#define cudaErrorCheck(ans){ cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr,"CUDA Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

template void AllocateDeviceField<float>(float**, size_t);
template void AllocateDeviceField<double>(double**, size_t);

template<typename T>
void AllocateDeviceField(T** d_array, size_t ArraySize)
{
	cudaErrorCheck(cudaMalloc(d_array, ArraySize));
}

template void FreeDeviceField<float>(float*);
template void FreeDeviceField<double>(double*);
template<typename T>
void FreeDeviceField(T* d_array)
{
	cudaErrorCheck(cudaFree(d_array));
}
