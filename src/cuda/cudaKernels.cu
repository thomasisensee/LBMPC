#include <stdio.h>

#include "cudaKernels.cuh"
#include "cudaErrorHandler.cuh"

#include "core/descriptors/latticeDescriptors.h"
#include "cell.h"
#include "core/constants.h"
#include "core/lb/lbConstants.h"
#include "core/kernelParameters.h"


__device__ unsigned int pos(unsigned int i, unsigned int j, unsigned int width) {
    return j * width + i;
}

template<typename T, unsigned int D, unsigned int Q>
__global__ void initializeDistributionsKernel(T* collision, const CollisionParamsBGK<T>* const params) {

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i > params->Nx - 1 || j > params->Ny - 1) { return; }
    
    unsigned int idx = pos(i, j, params->Nx);
    
    Cell<T,D,Q> cell;
    T R = 1.0;
    T U = 0.0;
    T V = 0.0;

    cell.setEquilibriumDistribution(&collision[Q * idx], R, U, V);
}

template<typename T, unsigned int D, unsigned int Q>
void initializeDistributionsCaller(T* deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize) {
    initializeDistributionsKernel<T,D,Q><<<gridSize, blockSize>>>(deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}
template void initializeDistributionsCaller<float,2,9>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void initializeDistributionsCaller<float,2,5>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void initializeDistributionsCaller<double,2,9>(double* deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);
template void initializeDistributionsCaller<double,2,5>(double* deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);

template<typename T, unsigned int D, unsigned int Q>
__global__ void doStreamingKernel(const T *const collision, T *streaming, const CollisionParamsBGK<T>* const params) {

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx = pos(i, j, params->Nx);
    unsigned int idxNeighbor;
    int cx, cy;

    for (unsigned int l = 0; l < Q; ++l) {
        cx = latticeDescriptors::latticeVelocities<D,Q>(l, 0);
        cy = latticeDescriptors::latticeVelocities<D,Q>(l, 1);

 	    idxNeighbor = pos(static_cast<int>(i) - cx, static_cast<int>(j) - cy, params->Nx);
		streaming[Q * idx + l] = collision[Q * idxNeighbor + l];
	}
}

template<typename T, unsigned int D, unsigned int Q>
void doStreamingCaller(T** deviceCollision, T** deviceStreaming, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize) {
    // Call the cuda kernel
    doStreamingKernel<T,D,Q><<<gridSize, blockSize>>>(*deviceCollision, *deviceStreaming, params);
    cudaErrorCheck(cudaDeviceSynchronize());

    // Swap the pointers
    T* swap = *deviceCollision;
    *deviceCollision = *deviceStreaming;
    *deviceStreaming = swap;
}
template void doStreamingCaller<float,2,9>(float** deviceCollision, float** deviceStreaming, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void doStreamingCaller<float,2,5>(float** deviceCollision, float** deviceStreaming, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void doStreamingCaller<double,2,9>(double** deviceCollision, double** deviceStreaming, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);
template void doStreamingCaller<double,2,5>(double** deviceCollision, double** deviceStreaming, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);

template<typename T, unsigned int D, unsigned int Q>
__global__ void doCollisionBGKKernel(T* collision, const CollisionParamsBGK<T>* const params) {

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx = pos(i, j, params->Nx);

    Cell<T,D,Q> cell;
    T R = cell.getZerothMoment(&collision[idx * Q]);
    T U = cell.getVelocityX(&collision[idx * Q], R);
    T V = cell.getVelocityY(&collision[idx * Q], R);

    cell.computePostCollisionDistributionBGK(&collision[idx * Q], params, R, U, V);
}


template<typename T, unsigned int D, unsigned int Q>
void doCollisionBGKCaller(T* deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize) {
    doCollisionBGKKernel<T,D,Q><<<gridSize, blockSize>>>(deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}
template void doCollisionBGKCaller<float,2,9>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionBGKCaller<float,2,5>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionBGKCaller<double,2,9>(double* deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionBGKCaller<double,2,5>(double* deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);


template<typename T, unsigned int D, unsigned int Q>
__global__ void doCollisionCHMKernel(T* collision, const CollisionParamsCHM<T>* const params) {

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx = pos(i, j, params->Nx);

    Cell<T,D,Q> cell;
    T R = cell.getZerothMoment(&collision[idx * Q]);
    T U = cell.getVelocityX(&collision[idx * Q], R);
    T V = cell.getVelocityY(&collision[idx * Q], R);

    cell.computePostCollisionDistributionCHM(&collision[idx * Q], params, R, U, V);
}


template<typename T, unsigned int D, unsigned int Q>
void doCollisionCHMCaller(T* deviceCollision, const CollisionParamsCHM<T>* const params, dim3 gridSize, dim3 blockSize) {
    doCollisionCHMKernel<T,D,Q><<<gridSize, blockSize>>>(deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}
template void doCollisionCHMCaller<float,2,9>(float* deviceCollision, const CollisionParamsCHM<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionCHMCaller<float,2,5>(float* deviceCollision, const CollisionParamsCHM<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionCHMCaller<double,2,9>(double* deviceCollision, const CollisionParamsCHM<double>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionCHMCaller<double,2,5>(double* deviceCollision, const CollisionParamsCHM<double>* const params, dim3 gridSize, dim3 blockSize);



template<typename T, unsigned int D, unsigned int Q>
__device__ void applyBC(T* collision, const BoundaryParams<T>* const params, unsigned int i, unsigned int j) {
    unsigned int idx, idxNeighbor, iPop, iPopRev;
    int cix, ciy;
    Cell<T,D,Q> cell;
    T R, dotProduct;

    idx = pos(i, j, params->Nx);

    for (unsigned int l = 0; l < 3; ++l) {
        iPop = latticeDescriptors::boundaryMapping<D,Q>(static_cast<unsigned int>(params->location), l);
        iPopRev = Q - iPop;
        cix = latticeDescriptors::latticeVelocities<D,Q>(iPop, 0);
        ciy = latticeDescriptors::latticeVelocities<D,Q>(iPop, 1);
/*
        if (params->location == BoundaryLocation::EAST) {
            printf("iPop = %d | (cix,ciy) = (%d,%d)\n",iPop,cix,ciy);
        }
*/
        // Check if the neighbor is outside the domain (i.e., a ghost cell)
        if (static_cast<int>(i) + cix < 1 || static_cast<int>(i) + cix > params->Nx - 2 || static_cast<int>(j) + ciy < 1 || static_cast<int>(j) + ciy > params->Ny - 2) { continue; }

        // Compute the index of the neighbor
        idxNeighbor = pos(static_cast<int>(i) + cix, static_cast<int>(j) + ciy, params->Nx);

        // Compute the dot product if WALL_VELOCITY is not null
        if (params->WALL_VELOCITY != nullptr) {
            dotProduct = static_cast<T>(cix) * params->WALL_VELOCITY[0] + static_cast<T>(ciy) * params->WALL_VELOCITY[1];
        } else {
            dotProduct = 0.0;
        }

        // Compute the zeroth moment (i.e., the density) of the neighbor cell
        R = cell.getZerothMoment(&collision[idxNeighbor * Q]);

        // Apply the bounce-back boundary condition
        collision[idx * Q + iPop] = collision[idxNeighbor * Q + iPopRev] + 2.0 * R * dotProduct * latticeDescriptors::latticeWeights<T,D,Q>(iPop) * latticeDescriptors::invCs2<T,D,Q>();
/*
        if (params->location == BoundaryLocation::NORTH) {
            printf("coll = %g, additional term = %g\n",collision[idxNeighbor * Q + iPopRev],2.0 * R * dotProduct * latticeDescriptors::latticeWeights<T,D,Q>(iPop) * latticeDescriptors::invCs2<T,D,Q>());
        }
        */
    }
       
}

template<typename T, unsigned int D, unsigned int Q>
__global__ void applyBounceBackKernel(T* collision, const BoundaryParams<T>* const params) {

    unsigned int i, j;

    switch(params->location) {
    case BoundaryLocation::WEST:
        i = 0;
        j = threadIdx.x + blockIdx.x * blockDim.x;
        if (j > params->Ny - 1) { return; }

        applyBC<T,D,Q>(collision, params, i, j);
        return;
    case BoundaryLocation::EAST:
        i = params->Nx-1;
        j = threadIdx.x + blockIdx.x * blockDim.x;
        if (j > params->Ny - 1) { return; }

        applyBC<T,D,Q>(collision, params, i, j);
        return;     
    case BoundaryLocation::SOUTH:
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = 0;
        if (i > params->Nx - 1) { return; }

        applyBC<T,D,Q>(collision, params, i, j);
        return;
    case BoundaryLocation::NORTH:
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = params->Ny-1;
        if (i > params->Nx - 1) { return; }
        applyBC<T,D,Q>(collision, params, i, j);
        return;
    }
}

template<typename T, unsigned int D, unsigned int Q>
void applyBounceBackCaller(T* deviceCollision, const BoundaryParams<T>* const params, dim3 gridSize, dim3 blockSize) {
    applyBounceBackKernel<T,D,Q><<<gridSize, blockSize>>>(deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}
template void applyBounceBackCaller<float,2,9>(float* deviceCollision, const BoundaryParams<float>* const params, dim3 gridSize, dim3 blockSize);
template void applyBounceBackCaller<float,2,5>(float* deviceCollision, const BoundaryParams<float>* const params, dim3 gridSize, dim3 blockSize);
template void applyBounceBackCaller<double,2,9>(double* deviceCollision, const BoundaryParams<double>* const params, dim3 gridSize, dim3 blockSize);
template void applyBounceBackCaller<double,2,5>(double* deviceCollision, const BoundaryParams<double>* const params, dim3 gridSize, dim3 blockSize);

template<typename T, unsigned int D, unsigned int Q>
__global__ void computeZerothMomentKernel(T* zerothMoment, const T* const collision, const CollisionParamsBGK<T>* const params) {

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx        = pos(i, j, params->Nx);
    unsigned int idxMoment  = pos(i - 1, j - 1, params->Nx - 2); // Since the zerothMoment array does not contain the ghost cells

    Cell<T,D,Q> cell;
    T R = cell.getZerothMoment(&collision[idx * Q]);
    zerothMoment[idxMoment] = R;
}

template<typename T, unsigned int D, unsigned int Q>
void computeZerothMomentCaller(T* deviceZerothMoment, const T* const deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize) {
    computeZerothMomentKernel<T,D,Q><<<gridSize, blockSize>>>(deviceZerothMoment, deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}
template void computeZerothMomentCaller<float,2,9>(float* deviceZerothMoment, const float* const deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void computeZerothMomentCaller<float,2,5>(float* deviceZerothMoment, const float* const deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void computeZerothMomentCaller<double,2,9>(double* deviceZerothMoment, const double* const deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);
template void computeZerothMomentCaller<double,2,5>(double* deviceZerothMoment, const double* const deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);

template<typename T, unsigned int D, unsigned int Q>
__global__ void computeFirstMomentKernel(T* firstMoment, const T* const collision, const CollisionParamsBGK<T>* const params) {

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx        = pos(i, j, params->Nx);
    unsigned int idxMoment  = pos(i - 1, j - 1, params->Nx - 2); // Since the zerothMoment array does not contain the ghost cells

    Cell<T,D,Q> cell;
    T U = cell.getFirstMomentX(&collision[idx * Q]);
    T V = cell.getFirstMomentY(&collision[idx * Q]);

    firstMoment[idxMoment * D]      = U;
    firstMoment[idxMoment * D + 1]  = V;
}

template<typename T, unsigned int D, unsigned int Q>
void computeFirstMomentCaller(T* deviceFirstMoment, const T* const deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize) {
    computeFirstMomentKernel<T,D,Q><<<gridSize, blockSize>>>(deviceFirstMoment, deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}
template void computeFirstMomentCaller<float,2,9>(float* deviceFirstMoment, const float* const deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void computeFirstMomentCaller<float,2,5>(float* deviceFirstMoment, const float* const deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void computeFirstMomentCaller<double,2,9>(double* deviceFirstMoment, const double* const deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);
template void computeFirstMomentCaller<double,2,5>(double* deviceFirstMoment, const double* const deviceCollision, const CollisionParamsBGK<double>* const params, dim3 gridSize, dim3 blockSize);