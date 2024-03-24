#ifndef LatticeGrid_HH
#define LatticeGrid_HH

#include <stdio.h>

#include "lbmGrid.h"
#include "cuda.h"

template<typename T>
LBMGrid<T>::LBMGrid(LBMModelWrapper<T>* lbmModel, GridGeometry2DWrapper<T>* gridGeometry, bool GPU) : lbmModel(lbmModel), gridGeometry(gridGeometry)
{
    GPU_ENABLED = GPU;
    unsigned int gridSize = lbmModel->getHostModel()->getQ() * gridGeometry->getHostGridGeometry()->getGhostVolume();
    collision = new T[gridSize];
}

template<typename T>
LBMGrid<T>::~LBMGrid()
{
    delete[] collision;
    if(GPU_ENABLED)
    {
        freeDeviceField<T>(collision);
        freeDeviceField<T>(d_streaming);
    }
}

template<typename T>
void LBMGrid<T>::initializeLBMDistributionsCPU(T* h_data)
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

template<typename T>
T* LBMGrid<T>::getDeviceCollisionPtr()
{
    return collision;
}

template<typename T>
T* LBMGrid<T>::getDeviceStreamingPtr()
{
    return d_streaming;
}

template<typename T>
LBMGridWrapper<T>::LBMGridWrapper(LBMGrid<T>* lbmGrid) : hostLBMGrid(lbmGrid), deviceLBMGrid(nullptr)
{
    allocateOnDevice();  
}

template<typename T>
LBMGridWrapper<T>::~LBMGridWrapper()
{
    if(deviceLBMGrid)
    {
        cudaFree(deviceLBMGrid->getDeviceCollisionPtr());
        cudaFree(deviceLBMGrid->getDeviceStreamingPtr());
        cudaFree(deviceLBMGrid);
    }
}

template<typename T>
void LBMGridWrapper<T>::allocateOnDevice()
{
    // Allocate device version of LBMGrid object and copy data
    cudaMalloc(&deviceLBMGrid, sizeof(LBMGrid<T>));
    cudaMemcpy(deviceLBMGrid, hostLBMGrid, sizeof(LBMGrid<T>), cudaMemcpyHostToDevice);

    // Allocate device memory for collision and streaming fields
    size_t gridSize = sizeof(T)*hostLBMGrid->gridGeometry->getHostGridGeometry()->getGhostVolume();
    T* CollisionPtr = deviceLBMGrid->getDeviceCollisionPtr();
    T* StreamingPtr = deviceLBMGrid->getDeviceCollisionPtr();
    cudaMalloc(&(CollisionPtr), gridSize);
    cudaMalloc(&(StreamingPtr), gridSize);
}

template<typename T>
LBMGrid<T>* LBMGridWrapper<T>::getHostGrid() const
{
    return hostLBMGrid;
}

template<typename T>
LBMGrid<T>* LBMGridWrapper<T>::getDeviceGrid() const
{
    return deviceLBMGrid;
}

#endif
