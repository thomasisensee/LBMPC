#ifndef GRID_GEOMETRY_1D_H
#define GRID_GEOMETRY_1D_H

#include "gridGeometryBase.h"

/*************************************/
/***** Derived grid geometry: 1D *****/
/*************************************/
template<typename T>
class GridGeometry1D : public GridGeometry<T> {
protected:
    /// Number of nodes in the direction x
    const unsigned int _nX;

public:
    /// Constructor
    GridGeometry1D(T delta, unsigned int nX);
    /// Read access to grid width
    unsigned int getNx() const;
    /// Read access to grid width
    unsigned int getGhostNx() const;
    /// Read access to grid volume
    unsigned int getVolume() const;
    /// Read access to grid volume
    unsigned int getGhostVolume() const;
    /// Prints grid details
    void printParameters() const;
};

#include "gridGeometry1D.hh"

#endif // GRID_GEOMETRY_1D_H