#ifndef GRID_GEOMETRY_H
#define GRID_GEOMETRY_H

/******************************/
/***** Base grid geometry *****/
/******************************/
template<typename T>
class GridGeometry {
protected:
    /// Distance to the next node
    const T _delta;

public:
    /// Constructor
    explicit GridGeometry(T delta);

    /// Read access to the distance of grid nodes
    T getDelta() const;

    /// One-dimensional mapping function
    __host__ __device__ static unsigned int pos(unsigned int i) {
        return i;
    };

    /// Two-dimensional mapping function
    __host__ __device__ static unsigned int pos(unsigned int i, unsigned int j, unsigned int width) {
        return j * width + i;
    };

    /// Three-dimensional mapping function
    __host__ __device__ static unsigned int pos(unsigned int i, unsigned int j, unsigned int k, unsigned int width, unsigned int height) {
        return (k * height + j) * width + i;
    };
};

/*************************************/
/***** Derived grid geometry: 2D *****/
/*************************************/
template<typename T>
class GridGeometry2D : public GridGeometry<T> {
protected:
    /// Number of nodes in the direction x and y
    const unsigned int _nX, _nY;

public:
    /// Constructor
    GridGeometry2D(T delta, unsigned int nX, unsigned int nY);
    /// Read access to grid width
    unsigned int getNx() const;
    /// Read access to grid height
    unsigned int getNy() const;
    /// Read access to grid width
    unsigned int getGhostNx() const;
    /// Read access to grid height
    unsigned int getGhostNy() const;
    /// Read access to grid volume
    unsigned int getVolume() const;
    /// Read access to grid volume
    unsigned int getGhostVolume() const;
    /// Prints grid details
    void printParameters() const;
};

#include "gridGeometry.hh"

#endif // GRID_GEOMETRY_H
