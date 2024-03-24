#ifndef GridGeometry_HH
#define GridGeometry_HH

#include <iostream>
#include "gridGeometry.h"

template<typename T>
GridGeometry2D<T>::GridGeometry2D(T globPosX, T globPosY, T delta, int nX, int nY)
{
  init(globPosX, globPosY, delta, nX, nY);
}

template<typename T>
void GridGeometry2D<T>::init(T globPosX, T globPosY, T delta, int nX, int nY)
{
    _globPosX = globPosX;
    _globPosY = globPosY;
    _delta    = delta;
    _nX       = nX;
    _nY       = nY;
}

template<typename T>
__host__ __device__ T GridGeometry2D<T>::getGlobPosX() const
{
    return _globPosX;
}

template<typename T>
__host__ __device__ T GridGeometry2D<T>::getGlobPosY() const
{
    return _globPosY;
}

template<typename T>
__host__ __device__ T GridGeometry2D<T>::getDelta() const
{
    return _delta;
}

template<typename T>
__host__ __device__ unsigned int GridGeometry2D<T>::getNx() const
{
    return _nX;
}

template<typename T>
__host__ __device__ unsigned int GridGeometry2D<T>::getNy() const
{
    return _nY;
}

template<typename T>
__host__ __device__ unsigned int GridGeometry2D<T>::getGhostNx() const
{
    return _nX+2;
}

template<typename T>
__host__ __device__ unsigned int GridGeometry2D<T>::getGhostNy() const
{
    return _nY+2;
}

template<typename T>
__host__ __device__ unsigned int GridGeometry2D<T>::getGhostVolume() const
{
    return (_nX+2)*(_nY+2);
}

template<typename T>
void GridGeometry2D<T>::print() const
{
    std::cout << "============== Grid Details ==============" << std::endl;
    std::cout << "==\tOrigin (x,y):" << "\t" << "(" << this->getGlobPosX() << "," << this->getGlobPosY() << ")" << "\t\t==" << std::endl;
    std::cout << "==\tExtent (Nx,Ny):"  << "\t" << "(" << this->getNx() << "/" << this->getNy() << ")" << "\t==" << std::endl;
    std::cout << "==\tΔx:" << "\t" << "\t" << this->getDelta() << "\t\t==" << std::endl;
    std::cout << "==========================================\n" << std::endl;
}

template<typename T>
GridGeometry2DWrapper<T>::GridGeometry2DWrapper(GridGeometry2D<T>* gridGeometry) : hostGridGeometry(gridGeometry), deviceGridGeometry(nullptr)
{
    allocateAndCopyToDevice();
}

template<typename T>
GridGeometry2DWrapper<T>::~GridGeometry2DWrapper()
{
    if(deviceGridGeometry)
    {
        cudaFree(deviceGridGeometry);
    }
}

template<typename T>
void GridGeometry2DWrapper<T>::allocateAndCopyToDevice()
{
    // Allocate device version of LBMModel object and copy data
    cudaMalloc(&deviceGridGeometry, sizeof(GridGeometry2D<T>));
    cudaMemcpy(deviceGridGeometry, hostGridGeometry, sizeof(GridGeometry2D<T>), cudaMemcpyHostToDevice);
}

template<typename T>
GridGeometry2D<T>* GridGeometry2DWrapper<T>::getHostGridGeometry() const
{
    return hostGridGeometry;
}

template<typename T>
GridGeometry2D<T>* GridGeometry2DWrapper<T>::getDeviceGridGeometry() const
{
    return deviceGridGeometry;
}
    
#endif
