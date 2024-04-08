#ifndef LB_GRID_HH
#define LB_GRID_HH

#include <stdio.h>
#include <cassert>
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
    if (_deviceStreaming != nullptr) {
        cudaErrorCheck(cudaFree(_deviceStreaming));
    }
}

template<typename T>
void LBGrid<T>::printParameters() {
    _gridGeometry->printParameters();
    _lbModel->printParameters();
    _collisionModel->printParameters();
    _boundaryConditionManager->printParameters();
}

template<typename T>
std::vector<T>& LBGrid<T>::getHostDistributions() {
    return _hostDistributions;
}

template<typename T>
std::vector<T>& LBGrid<T>::getHostZerothMoment() {
    return _hostZerothMoment;
}

template<typename T>
std::vector<T>& LBGrid<T>::getHostFirstMoment() {
    return _hostFirstMoment;
}

template<typename T>
T* LBGrid<T>::getDeviceCollision() const {
    return _deviceCollision;
}

template<typename T>
T* LBGrid<T>::getDeviceStreaming() const {
    return _deviceStreaming;
}

template<typename T>
T* LBGrid<T>::getDeviceZerothMoment() const {
    return _deviceZerothMoment;
}

template<typename T>
T* LBGrid<T>::getDeviceFirstMoment() const {
    return _deviceFirstMoment;
}

template<typename T>
const GridGeometry2D<T>& LBGrid<T>::getGridGeometry() const {
    assert(_gridGeometry != nullptr); // Optional: Ensure the unique_ptr is not empty
    return *_gridGeometry;
}

template<typename T>
void LBGrid<T>::allocateHostData() {
    _hostDistributions.resize(_lbModel->getQ() * _gridGeometry->getGhostVolume(), static_cast<T>(0));
    _hostZerothMoment.resize(_gridGeometry->getVolume(), static_cast<T>(0));
    _hostFirstMoment.resize(_lbModel->getD() * _gridGeometry->getVolume(), static_cast<T>(0));
}

template<typename T>
void LBGrid<T>::allocateDeviceData() {
    if(this->getDeviceCollision() == nullptr) {
        cudaErrorCheck(cudaMalloc(&_deviceCollision, _lbModel->getQ() * _gridGeometry->getGhostVolume() * sizeof(T)));
    }
    if(this->getDeviceStreaming() == nullptr) {
        cudaErrorCheck(cudaMalloc(&_deviceStreaming, _lbModel->getQ() * _gridGeometry->getGhostVolume() * sizeof(T)));
    }
    if(this->getDeviceZerothMoment() == nullptr) {
        cudaErrorCheck(cudaMalloc(&_deviceZerothMoment, _gridGeometry->getVolume() * sizeof(T)));
    }
    if(this->getDeviceFirstMoment() == nullptr) {
        cudaErrorCheck(cudaMalloc(&_deviceFirstMoment, _lbModel->getD() * _gridGeometry->getVolume() * sizeof(T)));
    }
}

template<typename T>
void LBGrid<T>::cleanupDevice() {
    if(this->getDeviceCollision() != nullptr) {
        cudaErrorCheck(cudaFree(this->getDeviceCollision()));
        _deviceCollision = nullptr;
    }
    if(this->getDeviceStreaming() != nullptr) {
        cudaErrorCheck(cudaFree(this->getDeviceStreaming()));
        _deviceStreaming = nullptr;
    }
    if(this->getDeviceZerothMoment() != nullptr) {
        cudaErrorCheck(cudaFree(this->getDeviceZerothMoment()));
        _deviceZerothMoment = nullptr;
    }
    if(this->getDeviceFirstMoment() != nullptr) {
        cudaErrorCheck(cudaFree(this->getDeviceFirstMoment()));
        _deviceFirstMoment = nullptr;
    }
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
void LBGrid<T>::fetchZerothMoment() {
    if(this->getDeviceZerothMoment() != nullptr) {
       cudaErrorCheck(cudaMemcpy(_hostZerothMoment.data(), _deviceZerothMoment, _gridGeometry->getVolume() * sizeof(T), cudaMemcpyDeviceToHost));
    }
}

template<typename T>
void LBGrid<T>::fetchFirstMoment() {
    if(this->getDeviceFirstMoment() != nullptr) {
       cudaErrorCheck(cudaMemcpy(_hostFirstMoment.data(), _deviceFirstMoment, _lbModel->getD() * _gridGeometry->getVolume() * sizeof(T), cudaMemcpyDeviceToHost));
    }
}


template<typename T>
void LBGrid<T>::copyToDevice() {
    if(this->getDeviceCollision() != nullptr) {
       cudaErrorCheck(cudaMemcpy(_deviceCollision, _hostDistributions.data(), _gridGeometry->getGhostVolume() * sizeof(T), cudaMemcpyHostToDevice));
    }
}

template<typename T>
void LBGrid<T>::copyToHost() {
    if(this->getDeviceStreaming() != nullptr) {
       cudaErrorCheck(cudaMemcpy(_hostDistributions.data(), _deviceCollision, _gridGeometry->getGhostVolume() * sizeof(T), cudaMemcpyDeviceToHost));
    }
}

template<typename T>
void LBGrid<T>::initializeDistributions() {
    dim3 blockSize(_threadsPerBlock.first, _threadsPerBlock.second);
    dim3 gridSize(_numBlocks.first, _numBlocks.first);

    initializeDistributionsCaller(_deviceCollision, _params.getDeviceParams(), gridSize, blockSize);
}

template<typename T>
void LBGrid<T>::performCollisionStep() {
    _collisionModel->doCollision(_deviceCollision, _numBlocks, _threadsPerBlock);
}

template<typename T>
void LBGrid<T>::performStreamingStep() {
    dim3 blockSize(_threadsPerBlock.first, _threadsPerBlock.second);
    dim3 gridSize(_numBlocks.first, _numBlocks.first);

    doStreamingCaller(&_deviceCollision, &_deviceStreaming, _params.getDeviceParams(), gridSize, blockSize);
}

template<typename T>
void LBGrid<T>::applyBoundaryConditions() {
    _boundaryConditionManager->apply(_deviceCollision);
}

template<typename T>
void LBGrid<T>::computeMoments() {
    computeZerothMoment();
    computeFirstMoment();
}

template<typename T>
void LBGrid<T>::computeZerothMoment() {
    dim3 blockSize(_threadsPerBlock.first, _threadsPerBlock.second);
    dim3 gridSize(_numBlocks.first, _numBlocks.first);

    computeZerothMomentCaller(_deviceZerothMoment, _deviceCollision, _params.getDeviceParams(), gridSize, blockSize);
    fetchZerothMoment();
}

template<typename T>
void LBGrid<T>::computeFirstMoment() {
    dim3 blockSize(_threadsPerBlock.first, _threadsPerBlock.second);
    dim3 gridSize(_numBlocks.first, _numBlocks.first);

    computeFirstMomentCaller(_deviceFirstMoment, _deviceCollision, _params.getDeviceParams(), gridSize, blockSize);
    fetchFirstMoment();
}

template<typename T>
unsigned int LBGrid<T>::pos(unsigned int i, unsigned int j, unsigned int Nx) {
    return j * Nx + i;
}

#endif // LB_GRID_HH
