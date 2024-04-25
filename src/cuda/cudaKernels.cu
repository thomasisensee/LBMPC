#include <stdio.h>
#include <tuple>
#include <type_traits>

#include "cudaKernels.cuh"
#include "cudaErrorHandler.cuh"

#include "core/descriptors/descriptors.h"
#include "core/descriptors/aliases.h"
#include "core/descriptors/fieldTags.h"
#include "core/functors/functors.h"
#include "cell.h"
#include "core/constants.h"
#include "core/kernelParameters.h"
#include "core/grid/gridGeometryBase.h"

/************************************/
/***** Initialize Distributions *****/
/************************************/
template<typename T,typename DESCRIPTOR>
__global__ void initializeDistributionsKernel(T* collision, const BaseParams* const params, T initialScalarValue) {
    // Local constants for easier access
    constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i > params->Nx - 1 || j > params->Ny - 1) { return; }
    
    unsigned int idx = GridGeometry2D<T>::pos(i, j, params->Nx);
    
    Cell<T,DESCRIPTOR> cell;
    T R = initialScalarValue;
    T U = 0.0;
    T V = 0.0;

    cell.setEquilibriumDistribution(&collision[Q * idx], R, U, V);
}

template<typename T,typename DESCRIPTOR>
void initializeDistributionsCaller(T* deviceCollision, const BaseParams* const params, T initialScalarValue, dim3 gridSize, dim3 blockSize) {
    initializeDistributionsKernel<T,DESCRIPTOR><<<gridSize, blockSize>>>(deviceCollision, params, initialScalarValue);
    cudaErrorCheck(cudaDeviceSynchronize());
}

/*********************/
/***** Streaming *****/
/*********************/
template<typename T,typename DESCRIPTOR>
__global__ void doStreamingKernel(const T *const collision, T *streaming, const BaseParams* const params) {
    // Local constants for easier access
    constexpr unsigned int D = DESCRIPTOR::LATTICE::D;
    constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx = GridGeometry2D<T>::pos(i, j, params->Nx);
    unsigned int idxNeighbor;
    int cx, cy;

    for (unsigned int l = 0; l < Q; ++l) {
        cx = descriptors::c<D,Q>(l, 0);
        cy = descriptors::c<D,Q>(l, 1);

 	    idxNeighbor = GridGeometry2D<T>::pos(static_cast<int>(i) - cx, static_cast<int>(j) - cy, params->Nx);
		streaming[Q * idx + l] = collision[Q * idxNeighbor + l];
	}
}

template<typename T,typename DESCRIPTOR>
void doStreamingCaller(T** deviceCollision, T** deviceStreaming, const BaseParams* const params, dim3 gridSize, dim3 blockSize) {
    // Call the cuda kernel
    doStreamingKernel<T,DESCRIPTOR><<<gridSize, blockSize>>>(*deviceCollision, *deviceStreaming, params);
    cudaErrorCheck(cudaDeviceSynchronize());

    // Swap the pointers
    T* swap = *deviceCollision;
    *deviceCollision = *deviceStreaming;
    *deviceStreaming = swap;
}

/**************************/
/***** Collision: BGK *****/
/**************************/
template<typename T,typename DESCRIPTOR, typename... FieldPtrs>
__global__ void doCollisionBGKKernel(T* collision, const CollisionParamsBGK<T>* const params, FieldPtrs... fields) {
    // Local constants for easier access
    constexpr unsigned int D = DESCRIPTOR::LATTICE::D;
    constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx = GridGeometry2D<T>::pos(i, j, params->Nx);

    T U, V;
    T* velocityField = nullptr;
    Cell<T,DESCRIPTOR> cell;
    T R = cell.getZerothMoment(&collision[idx * Q]);

    /*********************************************************************************************************/
    /****** Compile-time logic in order to determine if the velocity field is present in the descriptor ******/
    /****** This corresponds to the case where the velocity of the momentum conservation is used in the ******/
    /****** collision step of the energy conservation. If the velocity field is not present, the        ******/
    /****** velocity is computed from the distribution function.                                        ******/
    /*********************************************************************************************************/
    constexpr int isEnergyDistribution = Contains<descriptors::EnergyConservation, typename DESCRIPTOR::TYPE>::value;

    if constexpr (isEnergyDistribution) {
        constexpr int hasVelocityField = Contains<descriptors::VelocityField, typename DESCRIPTOR::FIELDS>::value;

        if constexpr (hasVelocityField) {
            constexpr int velocityIdx = IndexOf<descriptors::VelocityField, typename DESCRIPTOR::FIELDS>::value;
            velocityField = get_field<velocityIdx>(fields...);
            U = velocityField[idx * D];
            V = velocityField[idx * D + 1];
        } else if constexpr (hasVelocityField == 0) {
            U = 0.0;
            V = 0.0;
        }
    } else {
        U = cell.getVelocityX(&collision[idx * Q], R);
        V = cell.getVelocityY(&collision[idx * Q], R);
    }

    cell.computePostCollisionDistributionBGK(&collision[idx * Q], params, R, U, V);
}

template<typename T,typename DESCRIPTOR, typename... FieldPtrs,
    typename std::enable_if_t<(sizeof...(FieldPtrs) > 0), int>>
void doCollisionBGKCaller(T* deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize, FieldPtrs... fields) {
    doCollisionBGKKernel<T,DESCRIPTOR,FieldPtrs...><<<gridSize, blockSize>>>(deviceCollision, params, fields...);
    cudaErrorCheck(cudaDeviceSynchronize());
}

template<typename T,typename DESCRIPTOR>
void doCollisionBGKCaller(T* deviceCollision, const CollisionParamsBGK<T>* const params, dim3 gridSize, dim3 blockSize) {
    doCollisionBGKKernel<T,DESCRIPTOR,T*><<<gridSize, blockSize>>>(deviceCollision, params, nullptr);
    cudaErrorCheck(cudaDeviceSynchronize());
}

/**************************/
/***** Collision: CHM *****/
/**************************/
template<typename T,typename DESCRIPTOR>
__global__ void doCollisionCHMKernel(T* collision, const CollisionParamsCHM<T>* const params) {
    // Local constants for easier access
    constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx = GridGeometry2D<T>::pos(i, j, params->Nx);

    Cell<T,DESCRIPTOR> cell;
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

/*******************************/
/***** Boundary conditions *****/
/*******************************/
template<typename T,typename DESCRIPTOR,typename FUNCTOR>
__global__ void applyBoundaryConditionKernel(T* collision, const BoundaryParams* const params) {

    unsigned int i, j;

    FUNCTOR applyBC;

    switch(params->location) {
    case BoundaryLocation::WEST:
        i = 0;
        j = threadIdx.x + blockIdx.x * blockDim.x;
        if (j > params->Ny - 1) { return; }

        applyBC(collision, params, i, j);
        return;
    case BoundaryLocation::EAST:
        i = params->Nx-1;
        j = threadIdx.x + blockIdx.x * blockDim.x;
        if (j > params->Ny - 1) { return; }

        applyBC(collision, params, i, j);
        return;     
    case BoundaryLocation::SOUTH:
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = 0;
        if (i > params->Nx - 1) { return; }

        applyBC(collision, params, i, j);
        return;
    case BoundaryLocation::NORTH:
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = params->Ny-1;
        if (i > params->Nx - 1) { return; }

        applyBC(collision, params, i, j);
        return;
    }
}

template<typename T,typename DESCRIPTOR,typename FUNCTOR>
void applyBoundaryConditionCaller(T* deviceCollision, const BoundaryParams* const params, dim3 gridSize, dim3 blockSize) {
    applyBoundaryConditionKernel<T,DESCRIPTOR,FUNCTOR><<<gridSize, blockSize>>>(deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}

/***************************************/
/***** Moment computations: zeroth *****/
/***************************************/
template<typename T,typename DESCRIPTOR>
__global__ void computeZerothMomentKernel(T* zerothMoment, const T* const collision, const BaseParams* const params) {
    // Local constants for easier access
    constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx        = GridGeometry2D<T>::pos(i, j, params->Nx);
    unsigned int idxMoment  = GridGeometry2D<T>::pos(i - 1, j - 1, params->Nx - 2); // Since the zerothMoment array does not contain the ghost cells

    Cell<T,DESCRIPTOR> cell;
    T R = cell.getZerothMoment(&collision[idx * Q]);
    zerothMoment[idxMoment] = R;
}

template<typename T,typename DESCRIPTOR>
void computeZerothMomentCaller(T* deviceZerothMoment, const T* const deviceCollision, const BaseParams* const params, dim3 gridSize, dim3 blockSize) {
    computeZerothMomentKernel<T,DESCRIPTOR><<<gridSize, blockSize>>>(deviceZerothMoment, deviceCollision, params);
    cudaErrorCheck(cudaDeviceSynchronize());
}

/**************************************/
/***** Moment computations: First *****/
/**************************************/
template<typename T,typename DESCRIPTOR>
__global__ void computeFirstMomentKernel(T* firstMoment, const T* const collision, const BaseParams* const params, bool computeVelocity) {
    // Local constants for easier access
    constexpr unsigned int D = DESCRIPTOR::LATTICE::D;
    constexpr unsigned int Q = DESCRIPTOR::LATTICE::Q;

    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 1 || i > params->Nx - 2 || j < 1 || j > params->Ny - 2) { return; }

    unsigned int idx        = GridGeometry2D<T>::pos(i, j, params->Nx);
    unsigned int idxMoment  = GridGeometry2D<T>::pos(i - 1, j - 1, params->Nx - 2); // Since the zerothMoment array does not contain the ghost cells

    Cell<T,DESCRIPTOR> cell;
    T U, V;
    if (computeVelocity) {
        U = cell.getVelocityX(&collision[idx * Q]);
        V = cell.getVelocityY(&collision[idx * Q]);
    } else {
        U = cell.getFirstMomentX(&collision[idx * Q]);
        V = cell.getFirstMomentY(&collision[idx * Q]);
    }

    firstMoment[idxMoment * D]      = U;
    firstMoment[idxMoment * D + 1]  = V;
}

template<typename T,typename DESCRIPTOR>
void computeFirstMomentCaller(T* deviceFirstMoment, const T* const deviceCollision, const BaseParams* const params, bool computeVelocity, dim3 gridSize, dim3 blockSize) {
    computeFirstMomentKernel<T,DESCRIPTOR><<<gridSize, blockSize>>>(deviceFirstMoment, deviceCollision, params, computeVelocity);
    cudaErrorCheck(cudaDeviceSynchronize());
}



/********************************************/
/***** Explicit template instantiations *****/
/********************************************/
template void initializeDistributionsCaller<float,descriptors::StandardD2Q9<float>>(float* deviceCollision, const BaseParams* const params, float initialScalarValue, dim3 gridSize, dim3 blockSize);
template void initializeDistributionsCaller<float,descriptors::ScalarD2Q9<float>>(float* deviceCollision, const BaseParams* const params, float initialScalarValue, dim3 gridSize, dim3 blockSize);
template void initializeDistributionsCaller<float,descriptors::ScalarD2Q5<float>>(float* deviceCollision, const BaseParams* const params, float initialScalarValue, dim3 gridSize, dim3 blockSize);

template void doStreamingCaller<float,descriptors::StandardD2Q9<float>>(float** deviceCollision, float** deviceStreaming, const BaseParams* const params, dim3 gridSize, dim3 blockSize);
template void doStreamingCaller<float,descriptors::ScalarD2Q9<float>>(float** deviceCollision, float** deviceStreaming, const BaseParams* const params, dim3 gridSize, dim3 blockSize);
template void doStreamingCaller<float,descriptors::ScalarD2Q5<float>>(float** deviceCollision, float** deviceStreaming, const BaseParams* const params, dim3 gridSize, dim3 blockSize);

template void doCollisionBGKCaller<float,descriptors::StandardD2Q9<float>>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionBGKCaller<float,descriptors::ScalarD2Q9<float>>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionBGKCaller<float,descriptors::ScalarD2Q5<float>>(float* deviceCollision, const CollisionParamsBGK<float>* const params, dim3 gridSize, dim3 blockSize);

template void doCollisionCHMCaller<float,descriptors::StandardD2Q9<float>>(float* deviceCollision, const CollisionParamsCHM<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionCHMCaller<float,descriptors::ScalarD2Q9<float>>(float* deviceCollision, const CollisionParamsCHM<float>* const params, dim3 gridSize, dim3 blockSize);
template void doCollisionCHMCaller<float,descriptors::ScalarD2Q5<float>>(float* deviceCollision, const CollisionParamsCHM<float>* const params, dim3 gridSize, dim3 blockSize);

template void applyBoundaryConditionCaller<float,descriptors::StandardD2Q9<float>,functors::boundary::BounceBack<float,descriptors::StandardD2Q9<float>>>(float* deviceCollision, const BoundaryParams* const params, dim3 gridSize, dim3 blockSize);
template void applyBoundaryConditionCaller<float,descriptors::StandardD2Q9<float>,functors::boundary::MovingWall<float,descriptors::StandardD2Q9<float>>>(float* deviceCollision, const BoundaryParams* const params, dim3 gridSize, dim3 blockSize);
template void applyBoundaryConditionCaller<float,descriptors::ScalarD2Q9<float>,functors::boundary::BounceBack<float,descriptors::ScalarD2Q9<float>>>(float* deviceCollision, const BoundaryParams* const params, dim3 gridSize, dim3 blockSize);
template void applyBoundaryConditionCaller<float,descriptors::ScalarD2Q9<float>,functors::boundary::AntiBounceBack<float,descriptors::ScalarD2Q9<float>>>(float* deviceCollision, const BoundaryParams* const params, dim3 gridSize, dim3 blockSize);
template void applyBoundaryConditionCaller<float,descriptors::ScalarD2Q5<float>,functors::boundary::BounceBack<float,descriptors::ScalarD2Q5<float>>>(float* deviceCollision, const BoundaryParams* const params, dim3 gridSize, dim3 blockSize);
template void applyBoundaryConditionCaller<float,descriptors::ScalarD2Q5<float>,functors::boundary::AntiBounceBack<float,descriptors::ScalarD2Q5<float>>>(float* deviceCollision, const BoundaryParams* const params, dim3 gridSize, dim3 blockSize);

template void computeZerothMomentCaller<float,descriptors::StandardD2Q9<float>>(float* deviceZerothMoment, const float* const deviceCollision, const BaseParams* const params, dim3 gridSize, dim3 blockSize);
template void computeZerothMomentCaller<float,descriptors::ScalarD2Q9<float>>(float* deviceZerothMoment, const float* const deviceCollision, const BaseParams* const params, dim3 gridSize, dim3 blockSize);
template void computeZerothMomentCaller<float,descriptors::ScalarD2Q5<float>>(float* deviceZerothMoment, const float* const deviceCollision, const BaseParams* const params, dim3 gridSize, dim3 blockSize);

template void computeFirstMomentCaller<float,descriptors::StandardD2Q9<float>>(float* deviceFirstMoment, const float* const deviceCollision, const BaseParams* const params, bool computeVelocity, dim3 gridSize, dim3 blockSize);
template void computeFirstMomentCaller<float,descriptors::ScalarD2Q9<float>>(float* deviceFirstMoment, const float* const deviceCollision, const BaseParams* const params, bool computeVelocity, dim3 gridSize, dim3 blockSize);
template void computeFirstMomentCaller<float,descriptors::ScalarD2Q5<float>>(float* deviceFirstMoment, const float* const deviceCollision, const BaseParams* const params, bool computeVelocity, dim3 gridSize, dim3 blockSize);
