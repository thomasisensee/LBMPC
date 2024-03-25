#ifndef LatticeGrid_HH
#define LatticeGrid_HH

#include <stdio.h>

#include "lbmGrid.h"
#include "cuda/cudaKernels.h"
#include "cuda/cudaErrorHandler.h"

template<typename T>
LBMGrid<T>::LBMGrid(LBMModel<T>* lbmModel, GridGeometry2D<T>* gridGeometry) : lbmModel(lbmModel), gridGeometry(gridGeometry)
{
    unsigned int gridSize = lbmModel->getQ() * gridGeometry->getGhostVolume();
    collision = new T[gridSize];
}

template<typename T>
LBMGrid<T>::~LBMGrid()
{
    delete[] collision;
    //cudaErrorCheck(cudaFree(collision));
    //cudaErrorCheck(cudaFree(streaming));
}

template<typename T>
T* LBMGrid<T>::getDeviceCollisionPtr()
{
    return collision;
}

template<typename T>
T* LBMGrid<T>::getDeviceStreamingPtr()
{
    return streaming;
}

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
