#include <stdio.h>
#include <cuda_runtime.h>

#include "core/constants.h"
#include "core/kernelParameters.h"
#include "core/cell.h"
#include "cudaKernels.cuh"
#include "cudaErrorHandler.cuh"

__device__ unsigned int pos(unsigned int i, unsigned int j, unsigned int width) {
    return j * width + i;
}

template<typename T>
__global__ void initializeDistributionsKernel(T* collision, const LBParams<T>* const params) {

    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i>params->Nx || j > params->Ny) { return; }
    
    unsigned int idx = pos(i, j, params->Nx);
    
    Cell<T> cell;
    T R = 1;
    T U = 0;
    T V = 0;
    
    cell.setEquilibriumDistribution(&collision[params->Q*idx], params, R, U, V);
}

template<typename T>
void initializeDistributionsCaller(T* deviceCollision, const LBParams<T>* const params, dim3 gridSize, dim3 blockSize) {
    initializeDistributionsKernel<<<gridSize, blockSize>>>(deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}
template void initializeDistributionsCaller<float>(float* deviceCollision, const LBParams<float>* const params, dim3 gridSize, dim3 blockSize);
template void initializeDistributionsCaller<double>(double* deviceCollision, const LBParams<double>* const params, dim3 gridSize, dim3 blockSize);

template<typename T>
__global__ void doStreamingKernel(const T *const collision, T *streaming, const LBParams<T>* const params) {

    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i<1 || i>params->Nx-1 || j<1 || j > params->Ny-1) { return; }

    unsigned int idx = pos(i, j, params->Nx);
    unsigned int idxNeighbor;

    for (int l=0; l<params->Q; l++) {
	    idxNeighbor = pos(i-params->LATTICE_VELOCITIES[l*2],j-params->LATTICE_VELOCITIES[l*2+1],params->Nx);
		streaming[params->Q*idx+l]=collision[params->Q*idxNeighbor];
	}
}

template<typename T>
void doStreamingCaller(T* deviceCollision, T* deviceStreaming, T* swap, const LBParams<T>* const params, dim3 gridSize, dim3 blockSize) {
    doStreamingKernel<<<gridSize, blockSize>>>(deviceCollision, deviceStreaming, params);
    cudaErrorCheck(cudaDeviceSynchronize());
    swap = deviceCollision; deviceCollision = deviceStreaming; deviceStreaming = swap;
}
template void doStreamingCaller<float>(float* deviceCollision, float* deviceStreaming, float* swap, const LBParams<float>* const params, dim3 gridSize, dim3 blockSize);
template void doStreamingCaller<double>(double* deviceCollision, double* deviceStreaming, double* swap, const LBParams<double>* const params, dim3 gridSize, dim3 blockSize);


template<typename T>
__global__ void doCollisionCHMKernel(T* collision, const CollisionParamsCHM<T>* const params) {

    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i<1 || i>params->Nx-1 || j<1 || j > params->Ny-1) { return; }

    unsigned int idx = pos(i, j, params->Nx);

    Cell<T> cell;
    T R = cell.getZeroMoment(&collision[params->Q*idx], params);
    T U = cell.getVelocityX(&collision[params->Q*idx], params, R);
    T V = cell.getVelocityY(&collision[params->Q*idx], params, R);   

    cell.computePostCollisionDistributionCHM(&collision[params->Q*idx], params, R, U, V);
}

template<typename T>
void doCollisionCHMCaller(T* deviceCollision, const CollisionParamsCHM<T>* const params, dim3 gridSize, dim3 blockSize) {
    doCollisionCHMKernel<<<gridSize, blockSize>>>(deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}
template void doCollisionCHMCaller<float>(float* deviceCollision, const CollisionParamsCHM<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionCHMCaller<double>(double* deviceCollision, const CollisionParamsCHM<double>* const params, dim3 gridSize, dim3 blockSize);

template<typename T>
__device__ void applyBC(T* collision, const BoundaryParams<T>* const params, unsigned int i, unsigned int j) {
    unsigned int idx, idxNeighbor, pop, popRev;
    int cx, cy;
    Cell<T> cell;
    T R, dotProduct;

    idx = pos(i, j, params->Nx);
        
    for (int l = 0; l<3; ++l) {
        pop = params->POPULATION[l];
        popRev = params->OPPOSITE_POPULATION[pop];
        cx = params->LATTICE_VELOCITIES[pop*params->D];
        cy = params->LATTICE_VELOCITIES[pop*params->D+1];
        if ((int) i+cx < 0 || i+cx > params->Nx || (int) j+cy < 0 || j+cy > params->Ny) { continue; }
        idxNeighbor = pos(i+cx, j+cy, params->Nx);
        if (params->WALL_VELOCITY != nullptr) {
            dotProduct = cx*params->WALL_VELOCITY[0] + cy*params->WALL_VELOCITY[1];
        } else {
            dotProduct = 0.0;
        }
        R = cell.getZeroMoment(&collision[params->Q*idxNeighbor], params);
        printf("(%d,%d) -> (%d,%d) | pop = %d | popRev = %d | CX = %d, CY = %d |idxNeibor = %d | R = %g | dotProduct = %g\n",i,j,i+cx,j+cy,pop,popRev,params->LATTICE_VELOCITIES[pop*params->D],params->LATTICE_VELOCITIES[pop*params->D+1],idxNeighbor,R,dotProduct);
        
        collision[params->Q*idx + pop] = collision[params->Q*idxNeighbor + popRev] + 2.0*params->LATTICE_WEIGHTS[pop] * R * C_S_POW2_INV * dotProduct;
    }
       
}

template<typename T>
__global__ void applyBounceBackKernel(T* collision, const BoundaryParams<T>* const params) {

    unsigned int i, j;

    switch(params->location) {
    case BoundaryLocation::WEST:
        i = 0;
        j = threadIdx.x + blockIdx.x * blockDim.x;
        if (j > params->Ny-1) { return; }

        applyBC(collision, params, i, j);
        return;
        
        
        /*
    case BoundaryLocation::EAST:
        i = params->Nx+1;
        j = threadIdx.x + blockIdx.x * blockDim.x;
        if (j > params->Ny-1) { return; }
        
        idx = pos(i, j, params->Nx);
        return;
    case BoundaryLocation::SOUTH:
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = 0;
        if (i > params->Ny-1) { return; }
        
        idx = pos(i, j, params->Nx);
        return;
    case BoundaryLocation::NORTH:
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = params->Ny-1;
        if (i > params->Ny-1) { return; }

        idx = pos(i, j, params->Nx);
        return;
       */ 
    }

}

template<typename T>
void applyBounceBackCaller(T* deviceCollision, const BoundaryParams<T>* const params, dim3 gridSize, dim3 blockSize) {
    applyBounceBackKernel<<<gridSize, blockSize>>>(deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}
template void applyBounceBackCaller<float>(float* deviceCollision, const BoundaryParams<float>* const params, dim3 gridSize, dim3 blockSize);
template void applyBounceBackCaller<double>(double* deviceCollision, const BoundaryParams<double>* const params, dim3 gridSize, dim3 blockSize);

