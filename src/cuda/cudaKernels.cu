#include <stdio.h>
#include <cuda_runtime.h>

#include "core/constants.h"
#include "core/simulationParams.h"
#include "cudaKernels.cuh"
#include "cudaErrorHandler.cuh"

__device__ unsigned int pos(unsigned int i, unsigned int j, unsigned int width) {
    return j * width + i;
}

template<typename T>
__global__ void initializeDistributionsKernel(T* collision, const FluidParams<T>* const params) {

    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = pos(i, j, params->Nx);

    if(i>params->Nx || j > params->Ny) { return; }

	T firstOrder, secondOrder, thirdOrder, fourthOrder;
    int cix,ciy;
    T cixcs2,ciycs2;
    T U=0.0,V=0.0;

    for(int l=0; l<params->Q; l++) {
	    cix = params->LATTICE_VELOCITIES[l*2];
        ciy = params->LATTICE_VELOCITIES[l*2+1];
        cixcs2 = cix*cix-C_S_POW2;
        ciycs2 = ciy*ciy-C_S_POW2;
        firstOrder = C_S_POW2_INV*(U*cix+V*ciy);
        secondOrder = 0.5*C_S_POW4_INV*(cixcs2*U*U + ciycs2*V*V + 2.0*cix*ciy*U*V);
        thirdOrder = 0.5*C_S_POW6_INV*(cixcs2*ciy*U*U*V + ciycs2*cix*U*V*V);
        fourthOrder = 0.25*C_S_POW8_INV*(cixcs2*ciycs2*U*U*V*V);

        collision[params->Q*idx+l] = params->LATTICE_WEIGHTS[l]*(1.0 + firstOrder + secondOrder + thirdOrder + fourthOrder);
    }
}

template<typename T>
void initializeDistributionsCaller(T* deviceCollision, const FluidParams<T>* const params, dim3 gridSize, dim3 blockSize) {
    initializeDistributionsKernel<<<gridSize, blockSize>>>(deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}
template void initializeDistributionsCaller<float>(float* deviceCollision, const FluidParams<float>* const params, dim3 gridSize, dim3 blockSize);
template void initializeDistributionsCaller<double>(double* deviceCollision, const FluidParams<double>* const params, dim3 gridSize, dim3 blockSize);


template<typename T>
__global__ void doStreamingKernel(const T *const collision, T *streaming, const FluidParams<T>* const params) {

    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = pos(i, j, params->Nx);

    if(i>params->Nx || j > params->Ny) { return; }
    
    unsigned int idxNeighbor;
    for(int l=0; l<params->Q; l++) {
	    idxNeighbor = pos(i-params->LATTICE_VELOCITIES[l*2],j-params->LATTICE_VELOCITIES[l*2+1],params->Nx);
		streaming[params->Q*idx+l]=collision[params->Q*idxNeighbor];
	}
}

template<typename T>
void doStreaming(T* deviceCollision, T* deviceStreaming, const FluidParams<T>* const params, dim3 gridSize, dim3 blockSize) {
    doStreamingKernel<<<gridSize, blockSize>>>(deviceCollision, deviceStreaming, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}
template void doStreaming<float>(float* deviceCollision, float* deviceStreaming, const FluidParams<float>* const params, dim3 gridSize, dim3 blockSize);
template void doStreaming<double>(double* deviceCollision, double* deviceStreaming, const FluidParams<double>* const params, dim3 gridSize, dim3 blockSize);
