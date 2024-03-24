#ifndef LatticeGrid_H
#define LatticeGrid_H

#include <stdio.h>

#include "lbmModel.h"
#include "gridGeometry.h"
//#include "cudaKernels.h"

template<typename T>
class LBMGrid
{
private:
    bool GPU_ENABLED;
    /// Distribution functions f, i.e., a grid of cells
    T* collision=nullptr;
    T* d_streaming=nullptr;
    
public:
    /// LBM model providing dimensionality, velocity set, and weights
    LBMModelWrapper<T>* lbmModel;
    /// Grid geometry (spacing, Nx, Ny, etc.)
    GridGeometry2DWrapper<T>* gridGeometry;

    /// Constructor
    LBMGrid(LBMModelWrapper<T>* lbmModel,GridGeometry2DWrapper<T>* gridGeometry, bool GPU=true);
    /// Destructor
    ~LBMGrid();
    T* getDeviceCollisionPtr();
    T* getDeviceStreamingPtr();
    
    void initializeLBMDistributionsCPU(T* h_data);
    void initializeLBMDistributionsGPU(T* h_data);
};


/// Wrapper class for duplication on GPU
template<typename T>
class LBMGridWrapper {
private:
    /// Host-side LBMModel object
    LBMGrid<T>* hostLBMGrid;
    /// Device-side LBMModel object
    LBMGrid<T>* deviceLBMGrid;

public:
    // Constructor
    LBMGridWrapper(LBMGrid<T>* lbmGrid);

    // Destructor
    ~LBMGridWrapper();

    // Allocate device memory and copy data
    void allocateOnDevice();
    
    /// Get pointer to the host LBMModel object
    LBMGrid<T>* getHostGrid() const;
    
    /// Get pointer to the device LBMModel object
    LBMGrid<T>* getDeviceGrid() const;
};

#include "lbmGrid.hh"

#endif
