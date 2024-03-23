#ifndef GridGeometry_H
#define GridGeometry_H

#include <cuda_runtime.h>

template<typename T>
class GridGeometry2D
{

private:
    /// Global position of the left lower corner of the grid
    T _globPosX, _globPosY;
    /// Distance to the next node
    T _delta;
    /// Number of nodes in the direction x and y
    unsigned int _nX, _nY;

public:
    /// Construction of a grid
    GridGeometry2D(T globPosX, T globPosY, T delta, int nX, int nY);
    /// Initializes the grid
    void init(T globPosX, T globPosY, T delta, int nX, int nY);
    /// Read access to left lower corner coordinates
    __host__ __device__ T getGlobPosX() const;
    __host__ __device__ T getGlobPosY() const;
    /// Read access to the distance of grid nodes
    __host__ __device__ T getDelta() const;
    /// Read access to grid width
    __host__ __device__ unsigned int getNx() const;
    /// Read access to grid height
    __host__ __device__ unsigned int getNy() const;
    /// Read access to grid width
    __host__ __device__ unsigned int getGhostNx() const;
    /// Read access to grid height
    __host__ __device__ unsigned int getGhostNy() const;
    /// Read access to grid volume
    __host__ __device__ unsigned int getGhostVolume() const;
    /// Prints grid details
    void print() const;
};

#include "gridGeometry.hh"

#endif
