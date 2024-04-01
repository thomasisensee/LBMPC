#ifndef LATTICE_GRID_H
#define LATTICE_GRID_H

#include <stdio.h>
#include <vector>

#include "lbModel.h"
#include "gridGeometry.h"
#include "kernelParameters.h"

template<typename T>
class LBGrid {
private:
    /// Required model objects
    std::unique_ptr<LBModel<T>> _lbModel;
    std::unique_ptr<CollisionModel<T>> _collisionModel;
    std::unique_ptr<GridGeometry2D<T>> _gridGeometry;
    std::unique_ptr<BoundaryConditionManager<T>> _boundaryConditionManager;

    /// Distribution function
    std::vector<T> _hostDistributions;
    T* _deviceCollision = nullptr;
    T* _deviceStreaming = nullptr;

    /// swap pointer for switching streaming and collision device arrays
    T* _swap = nullptr;

    /// Parameters to pass to cuda kernels
    LBParamsWrapper<T> _params;

    /// Cuda grid and block size
    std::pair<unsigned int, unsigned int> _threadsPerBlock;
    std::pair<unsigned int, unsigned int> _numBlocks;

public:
    /// Constructor
    LBGrid(
        std::unique_ptr<LBModel<T>>&& model,
        std::unique_ptr<CollisionModel<T>>&& collision, 
        std::unique_ptr<GridGeometry2D<T>>&& geometry, 
        std::unique_ptr<BoundaryConditionManager<T>>&& boundary
    );

    /// Destructor
    ~LBGrid();

    void allocateHostData();
    void allocateDeviceData();
    void prepareKernelParams();
    void copyKernelParamsToDevice();
    void initializeDistributions();
    void copyToDevice();
    void copyToHost();
    void performCollisionStep();
    void performStreamingStep();
    void applyBoundaryConditions();
    static unsigned int pos(unsigned int i, unsigned int j, unsigned int Nx);
};

#include "lbGrid.hh"

#endif // LB_GRID_H
