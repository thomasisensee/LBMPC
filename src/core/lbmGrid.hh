#ifndef LatticeGrid_HH
#define LatticeGrid_HH

#include <stdio.h>
#include <cuda_runtime.h>

#include "lbmModel.h"
#include "lbmGrid.h"
#include "cuda/cudaConstants.cuh"
#include "cuda/cudaKernels.cuh"
#include "cuda/cudaErrorHandler.cuh"

template<typename T>
LBMGrid<T>::LBMGrid(
        std::unique_ptr<LBMModel<T>>&& model,
        std::unique_ptr<CollisionModel<T>>&& collision,
        std::unique_ptr<GridGeometry2D<T>>&& geometry,
        std::unique_ptr<BoundaryConditionManager<T>>&& boundary
) : lbmModel(std::move(model)), collisionModel(std::move(collision)), gridGeometry(std::move(geometry)), boundaryConditionManager(std::move(boundary)) {
    prepareKernelParams();
    copyKernelParamsToDevice();
    allocateHostData();
    allocateDeviceData();
    initializeDistributions();
}

template<typename T>
LBMGrid<T>::~LBMGrid() {
    if (deviceCollision != nullptr) {
        cudaErrorCheck(cudaFree(deviceCollision));
    }
    if (deviceCollision != nullptr) {
        cudaErrorCheck(cudaFree(deviceStreaming));
    }
    if (deviceParams != nullptr) {
        cudaErrorCheck(cudaFree(deviceParams));
    }
}

template<typename T>
void LBMGrid<T>::allocateHostData() {
    hostDistributions.resize(lbmModel->getQ() * gridGeometry->getGhostVolume(), static_cast<T>(0));
}

template<typename T>
void LBMGrid<T>::allocateDeviceData() {
    cudaErrorCheck(cudaMalloc(&deviceCollision, lbmModel->getQ() * gridGeometry->getGhostVolume() * sizeof(T)));
    cudaErrorCheck(cudaMalloc(&deviceStreaming, lbmModel->getQ() * gridGeometry->getGhostVolume() * sizeof(T)));
}

template<typename T>
void LBMGrid<T>::prepareKernelParams() {
    // Own kernel parameters
    this->hostParams.D = this->lbmModel->getQ();
    this->hostParams.Nx = this->gridGeometry->getGhostNx();
    this->hostParams.Ny = this->gridGeometry->getGhostNy();
    this->hostParams.Q = this->lbmModel->getQ();
    this->hostParams.LATTICE_VELOCITIES = this->lbmModel->getLatticeVelocitiesPtr();
    this->hostParams.LATTICE_WEIGHTS = this->lbmModel->getLatticeWeightsPtr();

    // Kernel parameters of collision model
    this->collisionModel->prepareKernelParams(&(this->hostParams));
    this->collisionModel->copyKernelParamsToDevice();
}

template<typename T>
void LBMGrid<T>::copyKernelParamsToDevice() {
    // Allocate device memory for lattice velocities and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = lbmModel->getQ() * lbmModel->getD() * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = lbmModel->getQ() * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));

    // Prepare the host-side copy of LBMParams with device pointers
    LBMParams<T> paramsTemp = hostParams; // Use a temporary host copy
    paramsTemp.LATTICE_VELOCITIES = deviceLatticeVelocities;
    paramsTemp.LATTICE_WEIGHTS = deviceLatticeWeights;

    // Allocate memory for the LBMParams struct on the device if not already allocated
    if (deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&deviceParams, sizeof(LBMParams<T>)));
    }

    // Copy the prepared LBMParams (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(deviceParams, &paramsTemp, sizeof(LBMParams<T>), cudaMemcpyHostToDevice));
}

template<typename T>
void LBMGrid<T>::initializeDistributions() {
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIMENSION, THREADS_PER_BLOCK_DIMENSION);
    dim3 numBlocks((gridGeometry->getGhostNx() + threadsPerBlock.x - 1) / threadsPerBlock.x, (gridGeometry->getGhostNy() + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeDistributionsCaller(deviceCollision, deviceParams, numBlocks, threadsPerBlock);
}

template<typename T>
void LBMGrid<T>::copyToDevice() {
       cudaErrorCheck(cudaMemcpy(deviceCollision, hostDistributions, gridGeometry->getGhostVolume() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void LBMGrid<T>::copyToHost() {
       cudaErrorCheck(cudaMemcpy(hostDistributions, deviceCollision, gridGeometry->getGhostVolume() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void LBMGrid<T>::performCollisionStep() {
    this->collisionModel->doCollision(deviceCollision);
}

template<typename T>
void LBMGrid<T>::performStreamingStep() {
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIMENSION, THREADS_PER_BLOCK_DIMENSION);
    dim3 numBlocks((gridGeometry->getGhostNx() + threadsPerBlock.x - 1) / threadsPerBlock.x, (gridGeometry->getGhostNy() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    doStreamingCaller(deviceCollision, deviceStreaming, swap, deviceParams, numBlocks, threadsPerBlock);
}

template<typename T>
void LBMGrid<T>::applyBoundaryConditions() {
    this->boundaryConditionManager->apply(deviceCollision);
}


template<typename T>
unsigned int LBMGrid<T>::pos(unsigned int i, unsigned int j, unsigned int Nx) {
    return j * Nx + i;
}

#endif
