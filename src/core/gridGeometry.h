#ifndef GridGeometry_H
#define GridGeometry_H

template<typename T>
class GridGeometry2D
{

private:
    /// Global position of the left lower corner of the grid
    T _globPosX, _globPosY;
    /// Number of nodes in the direction x and y
    unsigned int _nX, _nY;
    /// Distance to the next node
    T _delta;
public:
    /// Construction of a grid
    GridGeometry2D(T globPosX, T globPosY, T delta, int nX, int nY);
    /// Initializes the grid
    void init(T globPosX, T globPosY, T delta, int nX, int nY);
    /// Read access to left lower corner coordinates
    T getGlobPosX() const;
    T getGlobPosY() const;
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
    unsigned int getGhostVolume() const;
    /// Prints grid details
    void print() const;
};

#include "gridGeometry.hh"

#endif
