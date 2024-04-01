#ifndef LB_GRID_HH
#define LB_GRID_HH

#include <stdio.h>
#include <cuda_runtime.h>

#include "lbModel.h"
#include "lbGrid.h"
#include "cuda/cudaConstants.cuh"
#include "cuda/cudaKernels.cuh"
#include "cuda/cudaErrorHandler.cuh"

template<typename T>
LBGrid<T>::LBGrid(
        std::unique_ptr<LBModel<T>>&& model,
        std::unique_ptr<CollisionModel<T>>&& collision,
        std::unique_ptr<GridGeometry2D<T>>&& geometry,
        std::unique_ptr<BoundaryConditionManager<T>>&& boundary
) : _lbModel(std::move(model)), _collisionModel(std::move(collision)), _gridGeometry(std::move(geometry)), _boundaryConditionManager(std::move(boundary)) {
    prepareKernelParams();
    copyKernelParamsToDevice();
    allocateHostData();
    allocateDeviceData();
    initializeDistributions();
}

template<typename T>
LBGrid<T>::~LBGrid() {
    if (_deviceCollision != nullptr) {
        cudaErrorCheck(cudaFree(_deviceCollision));
    }
    if (_deviceCollision != nullptr) {
        cudaErrorCheck(cudaFree(_deviceStreaming));
    }
    if (_deviceParams != nullptr) {
        cudaErrorCheck(cudaFree(_deviceParams));
    }
}

template<typename T>
void LBGrid<T>::allocateHostData() {
    _hostDistributions.resize(_lbModel->getQ() * _gridGeometry->getGhostVolume(), static_cast<T>(0));
}

template<typename T>
void LBGrid<T>::allocateDeviceData() {
    cudaErrorCheck(cudaMalloc(&_deviceCollision, _lbModel->getQ() * _gridGeometry->getGhostVolume() * sizeof(T)));
    cudaErrorCheck(cudaMalloc(&_deviceStreaming, _lbModel->getQ() * _gridGeometry->getGhostVolume() * sizeof(T)));
}

template<typename T>
void LBGrid<T>::prepareKernelParams() {
    // Own kernel parameters
    _hostParams.D = _lbModel->getQ();
    _hostParams.Nx = _gridGeometry->getGhostNx();
    _hostParams.Ny = _gridGeometry->getGhostNy();
    _hostParams.Q = _lbModel->getQ();
    _hostParams.LATTICE_VELOCITIES = _lbModel->getLatticeVelocitiesPtr();
    _hostParams.LATTICE_WEIGHTS = _lbModel->getLatticeWeightsPtr();

    // Set block and grid size for cuda kernel execution
    _threadsPerBlock = std::make_pair(THREADS_PER_BLOCK_DIMENSION, THREADS_PER_BLOCK_DIMENSION);
    _numBlocks = std::make_pair(
        (_gridGeometry->getGhostNx() + _threadsPerBlock.first  - 1) / _threadsPerBlock.first,
        (_gridGeometry->getGhostNy() + _threadsPerBlock.second - 1) / _threadsPerBlock.second
    );

    // Kernel parameters of collision model
    _collisionModel->prepareKernelParams(&(_hostParams));
    printf("1\n");
    _collisionModel->copyKernelParamsToDevice();
    printf("2\n");

    // Kernel parameters of boundary conditions
    _boundaryConditionManager->prepareKernelParamsAndCopyToDevice(&(_hostParams), _lbModel.get());
}

template<typename T>
void LBGrid<T>::copyKernelParamsToDevice() {
    // Allocate device memory for lattice velocities and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = _lbModel->getQ() * _lbModel->getD() * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, _hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = _lbModel->getQ() * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, _hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));

    // Prepare the host-side copy of LBParams with device pointers
    LBParams<T> paramsTemp = _hostParams; // Use a temporary host copy
    paramsTemp.LATTICE_VELOCITIES = deviceLatticeVelocities;
    paramsTemp.LATTICE_WEIGHTS = deviceLatticeWeights;

    // Allocate memory for the LBParams struct on the device if not already allocated
    if (_deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&_deviceParams, sizeof(LBParams<T>)));
    }

    // Copy the prepared LBParams (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(_deviceParams, &paramsTemp, sizeof(LBParams<T>), cudaMemcpyHostToDevice));
}

template<typename T>
void LBGrid<T>::initializeDistributions() {
    dim3 blockSize(_threadsPerBlock.first, _threadsPerBlock.second);
    dim3 gridSize(_numBlocks.first, _numBlocks.first);

    initializeDistributionsCaller(_deviceCollision, _deviceParams, gridSize, blockSize);
}

template<typename T>
void LBGrid<T>::copyToDevice() {
       cudaErrorCheck(cudaMemcpy(_deviceCollision, _hostDistributions, _gridGeometry->getGhostVolume() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void LBGrid<T>::copyToHost() {
       cudaErrorCheck(cudaMemcpy(_hostDistributions, _deviceCollision, _gridGeometry->getGhostVolume() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void LBGrid<T>::performCollisionStep() {
    _collisionModel->doCollision(_deviceCollision, _numBlocks, _threadsPerBlock);
}

template<typename T>
void LBGrid<T>::performStreamingStep() {
    dim3 blockSize(_threadsPerBlock.first, _threadsPerBlock.second);
    dim3 gridSize(_numBlocks.first, _numBlocks.first);

    doStreamingCaller(_deviceCollision, _deviceStreaming, _swap, _deviceParams, gridSize, blockSize);
}

template<typename T>
void LBGrid<T>::applyBoundaryConditions() {
    _boundaryConditionManager->apply(_deviceCollision);
}


template<typename T>
unsigned int LBGrid<T>::pos(unsigned int i, unsigned int j, unsigned int Nx) {
    return j * Nx + i;
}

#endif // LB_GRID_HH
