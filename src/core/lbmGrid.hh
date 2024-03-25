#ifndef LatticeGrid_HH
#define LatticeGrid_HH

#include <stdio.h>

#include "lbmGrid.h"
#include "cuda/cudaKernels.h"
#include "cuda/cudaErrorHandler.h"

template<typename T, typename LBMModelWrapperClassType>
LBMGrid<T, LBMModelWrapperClassType>::LBMGrid(LBMModelWrapperClassType* lbmModel, GridGeometry2DWrapper<T>* gridGeometry, bool GPU) : lbmModel(lbmModel), gridGeometry(gridGeometry)
{
    GPU_ENABLED = GPU;
    unsigned int gridSize = lbmModel->getHostModel()->getQ() * gridGeometry->getHostGridGeometry()->getGhostVolume();
    collision = new T[gridSize];
}

template<typename T, typename LBMModelWrapperClassType>
LBMGrid<T, LBMModelWrapperClassType>::~LBMGrid()
{
    delete[] collision;
    if(GPU_ENABLED)
    {
        cudaErrorCheck(cudaFree(collision));
        cudaErrorCheck(cudaFree(d_streaming));
    }
}

template<typename T, typename LBMModelWrapperClassType>
void LBMGrid<T, LBMModelWrapperClassType>::initializeLBMDistributionsCPU(T* h_data)
{
#define pos(x,y)		(Nx*(y)+(x))

    unsigned int Q = lbmModel->getHostModel()->getQ();
    unsigned int Nx = gridGeometry->getHostGridGeometry()->getGhostNx();
    unsigned int Ny = gridGeometry->getHostGridGeometry()->getGhostNy();
	T firstOrder, secondOrder, thirdOrder, fourthOrder;

	for(int l=0; l<Q; l++)
	{
	    for(int i=0; i<Nx; i++)
	    {
	        for(int j=0; j<Ny; j++)
	        {
                firstOrder = 0.;
                secondOrder = 0.;
                thirdOrder = 0.;
                fourthOrder = 0.;
   		        h_data[Q*pos(i,j)+l] = lbmModel->getHostModel()->getWEIGHT(l)*(1. + firstOrder + secondOrder + thirdOrder + fourthOrder);
		    }
		}
	}

}

template<typename T, typename LBMModelWrapperClassType>
T* LBMGrid<T, LBMModelWrapperClassType>::getDeviceCollisionPtr()
{
    return collision;
}

template<typename T, typename LBMModelWrapperClassType>
T* LBMGrid<T, LBMModelWrapperClassType>::getDeviceStreamingPtr()
{
    return d_streaming;
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
