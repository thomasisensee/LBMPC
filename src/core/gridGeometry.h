#ifndef GRID_GEOMETRY_H
#define GRID_GEOMETRY_H

template<typename T>
class GridGeometry2D {
private:
    /// Number of nodes in the direction x and y
    const unsigned int _nX, _nY;
    /// Distance to the next node
    const T _delta;
public:
    /// Construction of a grid
    GridGeometry2D(T delta, unsigned int nX, unsigned int nY);
    /// Read access to the distance of grid nodes
    T getDelta() const;
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
    void print() const;
};

#include "gridGeometry.hh"

#endif // GRID_GEOMETRY_H
