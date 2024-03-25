#ifndef LatticeGrid_H
#define LatticeGrid_H

#include <stdio.h>

#include "lbmModel.h"
#include "gridGeometry.h"
//#include "cudaKernels.h"

template<typename T>
class LBMGrid {
private:
    std::unique_ptr<LBMModel<T>> model;
    std::unique_ptr<GridGeometry2D<T>> gridGeometry;
    std::unique_ptr<BoundaryConditionManager<T>> boundaryConditionManager;

    /// Distribution functions f
    T* collision;
    T* streaming;

public:
    /// Constructor
    LBMGrid(
        std::unique_ptr<LBMModel<T>>&& model, 
        std::unique_ptr<GridGeometry2D<T>>&& geometry, 
        std::unique_ptr<BoundaryConditionManager<T>>&& boundaryConditions
    );
};


/// Wrapper class for duplication on GPU
template<typename T, typename LBMGridClassType>
class LBMGridWrapper {
private:
    /// Host-side LBMModel object
    LBMGridClassType* hostLBMGrid;
    /// Device-side LBMModel object
    LBMGridClassType* deviceLBMGrid;

public:
    // Constructor
    LBMGridWrapper(LBMGridClassType* lbmGrid);

    // Destructor
    ~LBMGridWrapper();

    // Allocate device memory and copy data
    void allocateOnDevice();
    
    /// Get pointer to the host LBMModel object
    LBMGridClassType* getHostGrid() const;
    
    /// Get pointer to the device LBMModel object
    LBMGridClassType* getDeviceGrid() const;
};

#include "lbmGrid.hh"

#endif
