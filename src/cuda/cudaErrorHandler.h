#ifndef CUDA_ERROR_HANDLER_H
#define CUDA_ERROR_HANDLER_H

#include <stdio.h>
#include <stdexcept>

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

/*
void cudaCheckLastError()
{
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}
*/

#endif
