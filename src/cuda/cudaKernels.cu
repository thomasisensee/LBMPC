#include <stdio.h>
#include <tuple>
#include <type_traits>

#include "cudaKernels.cuh"
#include "cudaErrorHandler.cuh"

#include "core/descriptors/descriptors.h"
#include "core/descriptors/fieldTags.h"
#include "core/functors/functors.h"
#include "cell.h"
#include "core/constants.h"
#include "core/kernelParameters.h"


__device__ unsigned int pos(unsigned int i, unsigned int j, unsigned int width) {
    return j * width + i;
}

template<typename T,typename DESCRIPTOR>
__global__ void initializeDistributionsKernel(T* collision, const CollisionParamsBGK<T>* const params) {
    // Local constants for easier access
    constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i > params->Nx - 1 || j > params->Ny - 1) { return; }
    
    unsigned int idx = pos(i, j, params->Nx);
    
    Cell<T,typename DESCRIPTOR::LATTICE> cell;
    T R = 1.0;
    T U = 0.0;
    T V = 0.0;

    cell.setEquilibriumDistribution(&collision[Q * idx], R, U, V);
}

template<typename T,typename DESCRIPTOR>
void initializeDistributionsCaller(T* deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize) {
    initializeDistributionsKernel<T,DESCRIPTOR><<<gridSize, blockSize>>>(deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}

template<typename T,typename DESCRIPTOR>
__global__ void doStreamingKernel(const T *const collision, T *streaming, const CollisionParamsBGK<T>* const params) {
    // Local constants for easier access
    constexpr unsigned int D = DESCRIPTOR::LATTICE::D;
    constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx = pos(i, j, params->Nx);
    unsigned int idxNeighbor;
    int cx, cy;

    for (unsigned int l = 0; l < Q; ++l) {
        cx = descriptors::c<D,Q>(l, 0);
        cy = descriptors::c<D,Q>(l, 1);

 	    idxNeighbor = pos(static_cast<int>(i) - cx, static_cast<int>(j) - cy, params->Nx);
		streaming[Q * idx + l] = collision[Q * idxNeighbor + l];
	}
}

template<typename T,typename DESCRIPTOR>
void doStreamingCaller(T** deviceCollision, T** deviceStreaming, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize) {
    // Call the cuda kernel
    doStreamingKernel<T,DESCRIPTOR><<<gridSize, blockSize>>>(*deviceCollision, *deviceStreaming, params);
    cudaErrorCheck(cudaDeviceSynchronize());

    // Swap the pointers
    T* swap = *deviceCollision;
    *deviceCollision = *deviceStreaming;
    *deviceStreaming = swap;
}

template<typename T,typename DESCRIPTOR, typename... FieldPtrs>
__global__ void doCollisionBGKKernel(T* collision, const CollisionParamsBGK<T>* const params, FieldPtrs... fields) {
    // Local constants for easier access
    constexpr unsigned int D = DESCRIPTOR::LATTICE::D;
    constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx = pos(i, j, params->Nx);

    T U, V;
    T* velocityField = nullptr;
    Cell<T,typename DESCRIPTOR::LATTICE> cell;
    T R = cell.getZerothMoment(&collision[idx * Q]);

    /*********************************************************************************************************/
    /****** Compile-time logic in order to determine if the velocity field is present in the descriptor ******/
    /****** This corresponds to the case where the velocity of the momentum conservation is used in the ******/
    /****** collision step of the energy conservation. If the velocity field is not present, the        ******/
    /****** velocity is computed from the distribution function.                                        ******/
    /*********************************************************************************************************/
    constexpr int isMomentumDistribution = Contains<descriptors::MomentumConservation, typename DESCRIPTOR::TYPE>::value;

    if constexpr (isMomentumDistribution) {
        constexpr int hasVelocityField = Contains<descriptors::VelocityField, typename DESCRIPTOR::FIELDS>::value;

        if constexpr (hasVelocityField) {
            constexpr int velocityIdx = IndexOf<descriptors::VelocityField, typename DESCRIPTOR::FIELDS>::value;
            velocityField = get_field<velocityIdx>(fields...);
            U = velocityField[idx * D];
            V = velocityField[idx * D + 1];
        } else if constexpr (hasVelocityField == 0) {
            U = cell.getVelocityX(&collision[idx * Q], R);
            V = cell.getVelocityY(&collision[idx * Q], R);
        }
    } else {
        U = 0.0;
        V = 0.0;
    }

    cell.computePostCollisionDistributionBGK(&collision[idx * Q], params, R, U, V);
}


template<typename T,typename DESCRIPTOR, typename... FieldPtrs, typename std::enable_if_t<(sizeof...(FieldPtrs) > 0), int> = 0>
void doCollisionBGKCaller(T* deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize, FieldPtrs... fields) {
    doCollisionBGKKernel<T,DESCRIPTOR,FieldPtrs...><<<gridSize, blockSize>>>(deviceCollision, params, fields...);
    cudaErrorCheck(cudaDeviceSynchronize());
}

template<typename T,typename DESCRIPTOR>
void doCollisionBGKCaller(T* deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize) {
    doCollisionBGKKernel<T,DESCRIPTOR,T*><<<gridSize, blockSize>>>(deviceCollision, params, nullptr);
    cudaErrorCheck(cudaDeviceSynchronize());
}

template<typename T,typename DESCRIPTOR>
__global__ void doCollisionCHMKernel(T* collision, const CollisionParamsCHM<T>* const params) {
    // Local constants for easier access
    constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx = pos(i, j, params->Nx);

    Cell<T,typename DESCRIPTOR::LATTICE> cell;
    T R = cell.getZerothMoment(&collision[idx * Q]);
    T U = cell.getVelocityX(&collision[idx * Q], R);
    T V = cell.getVelocityY(&collision[idx * Q], R);

    cell.computePostCollisionDistributionCHM(&collision[idx * Q], params, R, U, V);
}


template<typename T,typename DESCRIPTOR>
void doCollisionCHMCaller(T* deviceCollision, const CollisionParamsCHM<T>* const params, dim3 gridSize, dim3 blockSize) {
    doCollisionCHMKernel<T,DESCRIPTOR><<<gridSize, blockSize>>>(deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}

template<typename T,typename DESCRIPTOR>
__device__ void applyBC(T* collision, const BoundaryParams<T>* const params, unsigned int i, unsigned int j) {
    // Local constants for easier access
    constexpr unsigned int D = DESCRIPTOR::LATTICE::D;
    constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

    unsigned int idx, idxNeighbor, iPop, iPopRev;
    int cix, ciy;
    Cell<T,typename DESCRIPTOR::LATTICE> cell;
    T R;

    idx = pos(i, j, params->Nx);

    for (unsigned int l = 0; l < 3; ++l) {
        iPop = descriptors::b<D,Q>(static_cast<unsigned int>(params->location), l);
        iPopRev = Q - iPop;
        cix = static_cast<T>(descriptors::c<D,Q>(iPop, 0));
        ciy = static_cast<T>(descriptors::c<D,Q>(iPop, 1));
        T uWall = 0.0, vWall = 0.0;
        T cixcs2 = 0.0, ciycs2 = 0.0;
        T firstOrder = 0.0, thirdOrder = 0.0;

        // Check if the neighbor is outside the domain (i.e., a ghost cell)
        if (static_cast<int>(i) + cix < 1 || static_cast<int>(i) + cix > params->Nx - 2 || static_cast<int>(j) + ciy < 1 || static_cast<int>(j) + ciy > params->Ny - 2) { continue; }

        // Compute the index of the neighbor
        idxNeighbor = pos(static_cast<int>(i) + cix, static_cast<int>(j) + ciy, params->Nx);

        // Compute the dot product if WALL_VELOCITY is not null
        if (params->WALL_VELOCITY != nullptr) {
            uWall = params->WALL_VELOCITY[0];
            vWall = params->WALL_VELOCITY[1];
            cixcs2 = cix * cix - cs2<T,D,Q>();
            ciycs2 = ciy * ciy - cs2<T,D,Q>();
            firstOrder = descriptors::invCs2<T,D,Q>() * (uWall * cix + vWall * ciy);
            thirdOrder = 0.5 * invCs2<T,D,Q>() * invCs2<T,D,Q>() * invCs2<T,D,Q>() * (cixcs2 * ciy * uWall * uWall * vWall + ciycs2 * cix * uWall * vWall * vWall);
        }

        // Compute the zeroth moment (i.e., the density) of the neighbor cell
        R = cell.getZerothMoment(&collision[idxNeighbor * Q]);

        // Apply the bounce-back boundary condition
        collision[idx * Q + iPop] = collision[idxNeighbor * Q + iPopRev] + 2.0 * R * descriptors::w<T,D,Q>(iPop) * (firstOrder + thirdOrder);
    }
       
}

template<typename T,typename DESCRIPTOR>
__global__ void applyBounceBackKernel(T* collision, const BoundaryParams<T>* const params) {

    unsigned int i, j;

    switch(params->location) {
    case BoundaryLocation::WEST:
        i = 0;
        j = threadIdx.x + blockIdx.x * blockDim.x;
        if (j > params->Ny - 1) { return; }

        applyBC<T,DESCRIPTOR>(collision, params, i, j);
        return;
    case BoundaryLocation::EAST:
        i = params->Nx-1;
        j = threadIdx.x + blockIdx.x * blockDim.x;
        if (j > params->Ny - 1) { return; }

        applyBC<T,DESCRIPTOR>(collision, params, i, j);
        return;     
    case BoundaryLocation::SOUTH:
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = 0;
        if (i > params->Nx - 1) { return; }

        applyBC<T,DESCRIPTOR>(collision, params, i, j);
        return;
    case BoundaryLocation::NORTH:
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = params->Ny-1;
        if (i > params->Nx - 1) { return; }
        applyBC<T,DESCRIPTOR>(collision, params, i, j);
        return;
    }
}

template<typename T,typename DESCRIPTOR>
void applyBounceBackCaller(T* deviceCollision, const BoundaryParams<T>* const params, dim3 gridSize, dim3 blockSize) {
    applyBounceBackKernel<T,DESCRIPTOR><<<gridSize, blockSize>>>(deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}

template<typename T,typename DESCRIPTOR>
__global__ void computeZerothMomentKernel(T* zerothMoment, const T* const collision, const CollisionParamsBGK<T>* const params) {
    // Local constants for easier access
    constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx        = pos(i, j, params->Nx);
    unsigned int idxMoment  = pos(i - 1, j - 1, params->Nx - 2); // Since the zerothMoment array does not contain the ghost cells

    Cell<T,typename DESCRIPTOR::LATTICE> cell;
    T R = cell.getZerothMoment(&collision[idx * Q]);
    zerothMoment[idxMoment] = R;
}

template<typename T,typename DESCRIPTOR>
void computeZerothMomentCaller(T* deviceZerothMoment, const T* const deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize) {
    computeZerothMomentKernel<T,DESCRIPTOR><<<gridSize, blockSize>>>(deviceZerothMoment, deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}

template<typename T,typename DESCRIPTOR>
__global__ void computeFirstMomentKernel(T* firstMoment, const T* const collision, const CollisionParamsBGK<T>* const params) {
    // Local constants for easier access
    constexpr unsigned int D = DESCRIPTOR::LATTICE::D;
    constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx        = pos(i, j, params->Nx);
    unsigned int idxMoment  = pos(i - 1, j - 1, params->Nx - 2); // Since the zerothMoment array does not contain the ghost cells

    Cell<T,typename DESCRIPTOR::LATTICE> cell;
    T U = cell.getFirstMomentX(&collision[idx * Q]);
    T V = cell.getFirstMomentY(&collision[idx * Q]);

    firstMoment[idxMoment * D]      = U;
    firstMoment[idxMoment * D + 1]  = V;
}

template<typename T,typename DESCRIPTOR>
void computeFirstMomentCaller(T* deviceFirstMoment, const T* const deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize) {
    computeFirstMomentKernel<T,DESCRIPTOR><<<gridSize, blockSize>>>(deviceFirstMoment, deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}



/********************************************/
/***** Explicit template instantiations *****/
/********************************************/
template void initializeDistributionsCaller<float,descriptors::StandardD2Q9<float>>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void initializeDistributionsCaller<float,descriptors::ScalarD2Q9<float>>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void initializeDistributionsCaller<float,descriptors::ScalarD2Q5<float>>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);

template void doStreamingCaller<float,descriptors::StandardD2Q9<float>>(float** deviceCollision, float** deviceStreaming, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void doStreamingCaller<float,descriptors::ScalarD2Q9<float>>(float** deviceCollision, float** deviceStreaming, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void doStreamingCaller<float,descriptors::ScalarD2Q5<float>>(float** deviceCollision, float** deviceStreaming, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);

template void doCollisionBGKCaller<float,descriptors::StandardD2Q9<float>>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionBGKCaller<float,descriptors::ScalarD2Q9<float>>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize, float* velocityField);
template void doCollisionBGKCaller<float,descriptors::ScalarD2Q5<float>>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize, float* velocityField);

template void doCollisionCHMCaller<float,descriptors::StandardD2Q9<float>>(float* deviceCollision, const CollisionParamsCHM<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionCHMCaller<float,descriptors::ScalarD2Q9<float>>(float* deviceCollision, const CollisionParamsCHM<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionCHMCaller<float,descriptors::ScalarD2Q5<float>>(float* deviceCollision, const CollisionParamsCHM<float>* const params, dim3 gridSize, dim3 blockSize);

template void applyBounceBackCaller<float,descriptors::StandardD2Q9<float>>(float* deviceCollision, const BoundaryParams<float>* const params, dim3 gridSize, dim3 blockSize);
template void applyBounceBackCaller<float,descriptors::ScalarD2Q9<float>>(float* deviceCollision, const BoundaryParams<float>* const params, dim3 gridSize, dim3 blockSize);
template void applyBounceBackCaller<float,descriptors::ScalarD2Q5<float>>(float* deviceCollision, const BoundaryParams<float>* const params, dim3 gridSize, dim3 blockSize);

template void computeZerothMomentCaller<float,descriptors::StandardD2Q9<float>>(float* deviceZerothMoment, const float* const deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void computeZerothMomentCaller<float,descriptors::ScalarD2Q9<float>>(float* deviceZerothMoment, const float* const deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void computeZerothMomentCaller<float,descriptors::ScalarD2Q5<float>>(float* deviceZerothMoment, const float* const deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);

template void computeFirstMomentCaller<float,descriptors::StandardD2Q9<float>>(float* deviceFirstMoment, const float* const deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void computeFirstMomentCaller<float,descriptors::ScalarD2Q9<float>>(float* deviceFirstMoment, const float* const deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void computeFirstMomentCaller<float,descriptors::ScalarD2Q5<float>>(float* deviceFirstMoment, const float* const deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
