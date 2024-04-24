#ifndef GRID_GEOMETRY_BASE_H
#define GRID_GEOMETRY_BASE_H

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

    /// Read access to grid volume
    virtual unsigned int getVolume() const = 0;

    /// Read access to grid volume
    virtual unsigned int getGhostVolume() const = 0;

    /// Prints grid details
    virtual void printParameters() const = 0;
};

#include "gridGeometryBase.hh"

#endif // GRID_GEOMETRY_BASE_H