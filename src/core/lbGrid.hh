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
}

// Getter for _hostDistributions
template<typename T>
std::vector<T>& LBGrid<T>::getHostDistributions() {
    return _hostDistributions;
}

template<typename T>
T* LBGrid<T>::getDeviceCollision() const {
    return _deviceCollision;
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
    // Own kernel parameters (and duplicate on device)
    _params.setValues(
        _lbModel->getD(),
        _gridGeometry->getGhostNx(),
        _gridGeometry->getGhostNy(),
        _lbModel->getQ(),
        _lbModel->getLatticeVelocitiesPtr(),
        _lbModel->getLatticeWeightsPtr()
    );

    // Set block and grid size for cuda kernel execution
    _threadsPerBlock = std::make_pair(THREADS_PER_BLOCK_DIMENSION, THREADS_PER_BLOCK_DIMENSION);
    _numBlocks = std::make_pair(
        (_gridGeometry->getGhostNx() + _threadsPerBlock.first  - 1) / _threadsPerBlock.first,
        (_gridGeometry->getGhostNy() + _threadsPerBlock.second - 1) / _threadsPerBlock.second
    );

    // Pass kernel parameters to collision model
    _collisionModel->prepareKernelParams(_params.getHostParams());

    // Pass kernel parameters to boundary conditions
    _boundaryConditionManager->prepareKernelParams(_params.getHostParams(), _lbModel.get());
}

template<typename T>
void LBGrid<T>::initializeDistributions() {
    dim3 blockSize(_threadsPerBlock.first, _threadsPerBlock.second);
    dim3 gridSize(_numBlocks.first, _numBlocks.first);

    initializeDistributionsCaller(_deviceCollision, _params.getDeviceParams(), gridSize, blockSize);
}

template<typename T>
void LBGrid<T>::copyToDevice() {
       cudaErrorCheck(cudaMemcpy(_deviceCollision, _hostDistributions.data(), _gridGeometry->getGhostVolume() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void LBGrid<T>::copyToHost() {
       cudaErrorCheck(cudaMemcpy(_hostDistributions.data(), _deviceCollision, _gridGeometry->getGhostVolume() * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void LBGrid<T>::performCollisionStep() {
    _collisionModel->doCollision(_deviceCollision, _numBlocks, _threadsPerBlock);
}

template<typename T>
void LBGrid<T>::performStreamingStep() {
    dim3 blockSize(_threadsPerBlock.first, _threadsPerBlock.second);
    dim3 gridSize(_numBlocks.first, _numBlocks.first);

    doStreamingCaller(_deviceCollision, _deviceStreaming, _swap, _params.getDeviceParams(), gridSize, blockSize);
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
