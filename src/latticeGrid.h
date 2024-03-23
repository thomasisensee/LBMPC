#ifndef LatticeGrid_H
#define LatticeGrid_H

#include <stdio.h>

#include "lbmModel.h"
#include "gridGeometry.h"
#include "cuda.h"

template<typename T>
class LatticeGrid
{
private:
    bool GPU_ENABLED;
    /// Distribution functions f, i.e., a grid of cells
    T* h_distribution; 
    T* d_collision=nullptr;
    T* d_streaming=nullptr;
    
public:
    /// LBM model providing dimensionality, velocity set, and weights
    LBMModel<T>* lbmModel;
    /// Grid geometry (spacing, Nx, Ny, etc.)
    GridGeometry2D<T>* gridGeometry;

    /// Constructor
    LatticeGrid(LBMModel<T>* model,GridGeometry2D<T>* grid,bool GPU=true);
    /// Destructor
    ~LatticeGrid();
    void initializeLBMDistributionsCPU(T* h_data);
};

#include "latticeGrid.hh"

#endif
