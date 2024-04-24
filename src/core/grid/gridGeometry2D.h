#ifndef GRID_GEOMETRY_2D_H
#define GRID_GEOMETRY_2D_H

#include "gridGeometry1D.h"

/*************************************/
/***** Derived grid geometry: 2D *****/
/*************************************/
template<typename T>
class GridGeometry2D : public GridGeometry1D<T> {
protected:
    /// Number of nodes in the direction x and y
    const unsigned int _nY;

public:
    /// Constructor
    GridGeometry2D(T delta, unsigned int nX, unsigned int nY);
    /// Read access to grid height
    unsigned int getNy() const;
    /// Read access to grid height
    unsigned int getGhostNy() const;
    /// Read access to grid volume
    unsigned int getVolume() const;
    /// Read access to grid volume
    unsigned int getGhostVolume() const;
    /// Prints grid details
    void printParameters() const;
};

#include "gridGeometry2D.hh"

#endif // GRID_GEOMETRY_2D_H
