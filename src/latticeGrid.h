#ifndef LatticeGrid_H
#define LatticeGrid_H

#include <stdio.h>

#include "lbmModel.h"
#include "gridGeometry.h"
#include "cell.h"

template<typename T, typename lbmModelType, typename gridGeometryType>
class LatticeGrid
{
private:
    /// LBM model providing dimensionality, velocity set, and weights
    lbmModelType *lbmModel;
    /// Grid geometry (spacing, Nx, Ny, etc.)
    gridGeometryType *gridGeometry;
    /// Distribution functions f, i.e., a grid of cells
    T *h_distribution; 
    T *d_collision=nullptr;
    T *d_streaming=nullptr;

public:
    /// Constructor
    LatticeGrid(lbmModelType* model,gridGeometryType* grid);
    ~LatticeGrid();
    void InitializeCPU(T *h_data);
    //void CollisionCPU();
    
};

#endif
