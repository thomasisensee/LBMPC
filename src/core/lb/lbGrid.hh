#ifndef LB_GRID_HH
#define LB_GRID_HH

#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>

#include "lbGrid.h"
#include "cuda/cudaConstants.cuh"
#include "cuda/cudaKernels.cuh"
#include "cuda/cudaErrorHandler.cuh"

template<typename T,typename DESCRIPTOR>
LBGrid<T,DESCRIPTOR>::LBGrid(
        std::unique_ptr<GridGeometry2D<T>>&& geometry,
        std::unique_ptr<CollisionModel<T,DESCRIPTOR>>&& collision,
        std::unique_ptr<BoundaryConditionManager<T,DESCRIPTOR>>&& boundary
) : _gridGeometry(std::move(geometry)), _collisionModel(std::move(collision)), _boundaryConditionManager(std::move(boundary)) {
    prepareKernelParams();
    allocateHostData();
    allocateDeviceData();
    initializeDistributions();
}

template<typename T,typename DESCRIPTOR>
LBGrid<T,DESCRIPTOR>::~LBGrid() {
    if (_deviceCollision != nullptr) {
        cudaErrorCheck(cudaFree(_deviceCollision));
    }
    if (_deviceStreaming != nullptr) {
        cudaErrorCheck(cudaFree(_deviceStreaming));
    }
}

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::printParameters() {
    _gridGeometry->printParameters();
    _collisionModel->printParameters();
    _boundaryConditionManager->printParameters();
}

template<typename T,typename DESCRIPTOR>
std::vector<T>& LBGrid<T,DESCRIPTOR>::getHostDistributions() {
    return _hostDistributions;
}

template<typename T,typename DESCRIPTOR>
std::vector<T>& LBGrid<T,DESCRIPTOR>::getHostZerothMoment() {
    return _hostZerothMoment;
}

template<typename T,typename DESCRIPTOR>
std::vector<T>& LBGrid<T,DESCRIPTOR>::getHostFirstMoment() {
    return _hostFirstMoment;
}

template<typename T,typename DESCRIPTOR>
T* LBGrid<T,DESCRIPTOR>::getDeviceCollision() const {
    return _deviceCollision;
}

template<typename T,typename DESCRIPTOR>
T* LBGrid<T,DESCRIPTOR>::getDeviceStreaming() const {
    return _deviceStreaming;
}

template<typename T,typename DESCRIPTOR>
T* LBGrid<T,DESCRIPTOR>::getDeviceZerothMoment() const {
    return _deviceZerothMoment;
}

template<typename T,typename DESCRIPTOR>
T* LBGrid<T,DESCRIPTOR>::getDeviceFirstMoment() const {
    return _deviceFirstMoment;
}

template<typename T,typename DESCRIPTOR>
const GridGeometry2D<T>& LBGrid<T,DESCRIPTOR>::getGridGeometry() const {
    assert(_gridGeometry != nullptr);
    return *_gridGeometry;
}

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::allocateHostData() {
    _hostDistributions.resize(DESCRIPTOR::LATTICE::Q * _gridGeometry->getGhostVolume(), static_cast<T>(0));
    _hostZerothMoment.resize(_gridGeometry->getVolume(), static_cast<T>(0));
    _hostFirstMoment.resize(DESCRIPTOR::LATTICE::D * _gridGeometry->getVolume(), static_cast<T>(0));
}

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::allocateDeviceData() {
    if(this->getDeviceCollision() == nullptr) {
        cudaErrorCheck(cudaMalloc(&_deviceCollision, DESCRIPTOR::LATTICE::Q * _gridGeometry->getGhostVolume() * sizeof(T)));
    }
    if(this->getDeviceStreaming() == nullptr) {
        cudaErrorCheck(cudaMalloc(&_deviceStreaming, DESCRIPTOR::LATTICE::Q * _gridGeometry->getGhostVolume() * sizeof(T)));
    }
    if(this->getDeviceZerothMoment() == nullptr) {
        cudaErrorCheck(cudaMalloc(&_deviceZerothMoment, _gridGeometry->getVolume() * sizeof(T)));
    }
    if(this->getDeviceFirstMoment() == nullptr) {
        cudaErrorCheck(cudaMalloc(&_deviceFirstMoment, DESCRIPTOR::LATTICE::D * _gridGeometry->getVolume() * sizeof(T)));
    }
}

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::cleanupDevice() {
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

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::prepareKernelParams() {
    // Own kernel parameters (and duplicate on device)
    _params.setValues(
        _gridGeometry->getGhostNx(),
        _gridGeometry->getGhostNy(),
        _collisionModel->getOmegaShear()
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
    _boundaryConditionManager->prepareKernelParams(_params.getHostParams());
}

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::fetchZerothMoment() {
    if(this->getDeviceZerothMoment() != nullptr) {
       cudaErrorCheck(cudaMemcpy(_hostZerothMoment.data(), _deviceZerothMoment, _gridGeometry->getVolume() * sizeof(T), cudaMemcpyDeviceToHost));
    }
}

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::fetchFirstMoment() {
    if(this->getDeviceFirstMoment() != nullptr) {
       cudaErrorCheck(cudaMemcpy(_hostFirstMoment.data(), _deviceFirstMoment, DESCRIPTOR::LATTICE::D * _gridGeometry->getVolume() * sizeof(T), cudaMemcpyDeviceToHost));
    }
}


template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::copyToDevice() {
    if(this->getDeviceCollision() != nullptr) {
       cudaErrorCheck(cudaMemcpy(_deviceCollision, _hostDistributions.data(), _gridGeometry->getGhostVolume() * sizeof(T), cudaMemcpyHostToDevice));
    }
}

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::copyToHost() {
    if(this->getDeviceStreaming() != nullptr) {
       cudaErrorCheck(cudaMemcpy(_hostDistributions.data(), _deviceCollision, _gridGeometry->getGhostVolume() * sizeof(T), cudaMemcpyDeviceToHost));
    }
}

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::initializeDistributions() {
    dim3 blockSize(_threadsPerBlock.first, _threadsPerBlock.second);
    dim3 gridSize(_numBlocks.first, _numBlocks.first);

    initializeDistributionsCaller<T,DESCRIPTOR>(_deviceCollision, _params.getDeviceParams(), gridSize, blockSize);
}

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::performCollisionStep() {
    _collisionModel->doCollision(_deviceCollision, _numBlocks, _threadsPerBlock);
}

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::performStreamingStep() {
    dim3 blockSize(_threadsPerBlock.first, _threadsPerBlock.second);
    dim3 gridSize(_numBlocks.first, _numBlocks.first);

    doStreamingCaller<T,DESCRIPTOR>(&_deviceCollision, &_deviceStreaming, _params.getDeviceParams(), gridSize, blockSize);
}

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::applyBoundaryConditions() {
    _boundaryConditionManager->apply(_deviceCollision);
}

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::computeMoments() {
    computeZerothMoment();
    computeFirstMoment();
}

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::computeZerothMoment() {
    dim3 blockSize(_threadsPerBlock.first, _threadsPerBlock.second);
    dim3 gridSize(_numBlocks.first, _numBlocks.first);

    computeZerothMomentCaller<T,DESCRIPTOR>(_deviceZerothMoment, _deviceCollision, _params.getDeviceParams(), gridSize, blockSize);
    fetchZerothMoment();
}

template<typename T,typename DESCRIPTOR>
void LBGrid<T,DESCRIPTOR>::computeFirstMoment() {
    dim3 blockSize(_threadsPerBlock.first, _threadsPerBlock.second);
    dim3 gridSize(_numBlocks.first, _numBlocks.first);

    computeFirstMomentCaller<T,DESCRIPTOR>(_deviceFirstMoment, _deviceCollision, _params.getDeviceParams(), gridSize, blockSize);
    fetchFirstMoment();
}

template<typename T,typename DESCRIPTOR>
unsigned int LBGrid<T,DESCRIPTOR>::pos(unsigned int i, unsigned int j, unsigned int Nx) {
    return j * Nx + i;
}

#endif // LB_GRID_HH