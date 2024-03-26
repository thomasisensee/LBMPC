#ifndef LatticeGrid_HH
#define LatticeGrid_HH

#include <stdio.h>
#include <cuda_runtime.h>

#include "lbmModel.h"
#include "lbmGrid.h"
#include "cuda/cudaKernels.cuh"
#include "cuda/cudaErrorHandler.cuh"

template<typename T>
LBMGrid<T>::LBMGrid(
        std::unique_ptr<LBMModel<T>>&& model,
        std::unique_ptr<CollisionModel<T>>&& collision,
        std::unique_ptr<GridGeometry2D<T>>&& geometry,
        std::unique_ptr<BoundaryConditionManager<T>>&& boundary
) : lbmModel(std::move(model)), collisionModel(std::move(collision)), gridGeometry(std::move(geometry)), boundaryConditionManager(std::move(boundary)) {
    prepareKernelParams(hostParams);
    copyKernelParamsToDevice(hostParams, deviceParams);
    allocateHostData();
    allocateDeviceData();
    initializeDistributions();
}

template<typename T>
LBMGrid<T>::~LBMGrid() {
   cudaErrorCheck(cudaFree(deviceCollision));
   cudaErrorCheck(cudaFree(deviceStreaming));
   cudaErrorCheck(cudaFree(deviceParams));
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
void LBMGrid<T>::prepareKernelParams(FluidParams<T> &paramsHost) {
    paramsHost.Nx = gridGeometry->getGhostNx();
    paramsHost.Ny = gridGeometry->getGhostNy();
    paramsHost.Q = lbmModel->getQ();
    paramsHost.omegaShear = 0.7;
    paramsHost.LATTICE_VELOCITIES = lbmModel->getLatticeVelocitiesPtr();
    paramsHost.LATTICE_WEIGHTS = lbmModel->getLatticeWeightsPtr();    
}

template<typename T>
void LBMGrid<T>::copyKernelParamsToDevice(const FluidParams<T> &hostParams, FluidParams<T>* &deviceParams) {

    // Allocate device memory for arrays and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = lbmModel->getQ() * lbmModel->getD() * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities,sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = lbmModel->getQ() * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights,sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));


    // Prepare a device-side KernelParams struct with device pointers
    FluidParams<T> deviceParamsTemp = hostParams; // Copy host params to a temp
    deviceParamsTemp.LATTICE_VELOCITIES = deviceLatticeVelocities; // Set to device pointer
    deviceParamsTemp.LATTICE_WEIGHTS = deviceLatticeWeights; // Set to device pointer

    // Allocate memory for the KernelParams struct on the device
    cudaErrorCheck(cudaMalloc(&deviceParams, sizeof(FluidParams<T>)));

    // Copy the prepared KernelParams (with device pointers) to the device
    cudaErrorCheck(cudaMemcpy(deviceParams, &deviceParamsTemp, sizeof(FluidParams<T>), cudaMemcpyHostToDevice));
    
    // Free obsolete memory
    cudaErrorCheck(cudaFree(deviceParamsTemp.LATTICE_VELOCITIES));
    cudaErrorCheck(cudaFree(deviceParamsTemp.LATTICE_WEIGHTS));
}

template<typename T>
void LBMGrid<T>::initializeDistributions() {

    const unsigned int threadsPerBlockSide = 16; // A common choice is to use a square block of 16x16 threads = 256 threads
    dim3 threadsPerBlock(threadsPerBlockSide, threadsPerBlockSide);
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

}

template<typename T>
void LBMGrid<T>::performStreamingStep() {

}

template<typename T>
unsigned int LBMGrid<T>::pos(unsigned int i, unsigned int j, unsigned int Nx) {
    return j * Nx + i;
}

#endif
