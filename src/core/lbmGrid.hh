#ifndef LatticeGrid_HH
#define LatticeGrid_HH

#include <stdio.h>

#include "lbmModel.h"
#include "lbmGrid.h"
#include "cuda/cudaKernels.h"
#include "cuda/cudaErrorHandler.h"

template<typename T>
LBMGrid<T>::LBMGrid(
        std::unique_ptr<LBMModel<T>>&& model, 
        std::unique_ptr<GridGeometry2D<T>>&& geometry, 
        std::unique_ptr<BoundaryConditionManager<T>>&& boundaryConditions
) : model(std::move(model)), gridGeometry(std::move(geometry)), boundaryConditionManager(std::move(boundaryConditions)) {}



template<typename T, typename LBMGridClassType>
LBMGridWrapper<T, LBMGridClassType>::LBMGridWrapper(LBMGridClassType* lbmGrid) : hostLBMGrid(lbmGrid), deviceLBMGrid(nullptr)
{
    allocateOnDevice();  
}

template<typename T, typename LBMGridClassType>
LBMGridWrapper<T, LBMGridClassType>::~LBMGridWrapper()
{
    if(deviceLBMGrid)
    {
        cudaErrorCheck(cudaFree(deviceLBMGrid->getDeviceCollisionPtr()));
        cudaErrorCheck(cudaFree(deviceLBMGrid->getDeviceStreamingPtr()));
        cudaErrorCheck(cudaFree(deviceLBMGrid));
    }
}

template<typename T, typename LBMGridClassType>
void LBMGridWrapper<T, LBMGridClassType>::allocateOnDevice()
{
    // Allocate device version of LBMGrid object and copy data
    cudaErrorCheck(cudaMalloc(&deviceLBMGrid, sizeof(LBMGridClassType)));
    cudaErrorCheck(cudaMemcpy(deviceLBMGrid, hostLBMGrid, sizeof(LBMGridClassType), cudaMemcpyHostToDevice));

    // Allocate device memory for collision and streaming fields
    size_t gridSize = sizeof(T)*hostLBMGrid->gridGeometry->getHostGridGeometry()->getGhostVolume();
    T* CollisionPtr = deviceLBMGrid->getDeviceCollisionPtr();
    T* StreamingPtr = deviceLBMGrid->getDeviceCollisionPtr();
    cudaErrorCheck(cudaMalloc(&(CollisionPtr), gridSize));
    cudaErrorCheck(cudaMalloc(&(StreamingPtr), gridSize));
}

template<typename T, typename LBMGridClassType>
LBMGridClassType* LBMGridWrapper<T, LBMGridClassType>::getHostGrid() const
{
    return hostLBMGrid;
}

template<typename T, typename LBMGridClassType>
LBMGridClassType* LBMGridWrapper<T, LBMGridClassType>::getDeviceGrid() const
{
    return deviceLBMGrid;
}

#endif
