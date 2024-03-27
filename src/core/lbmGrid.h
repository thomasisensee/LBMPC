#ifndef LATTICE_GRID_H
#define LATTICE_GRID_H

#include <stdio.h>
#include <vector>

#include "lbmModel.h"
#include "gridGeometry.h"
#include "kernelParams.h"

template<typename T>
class LBMGrid {
private:
    /// Required model objects
    std::unique_ptr<LBMModel<T>> lbmModel;
    std::unique_ptr<CollisionModel<T>> collisionModel;
    std::unique_ptr<GridGeometry2D<T>> gridGeometry;
    std::unique_ptr<BoundaryConditionManager<T>> boundaryConditionManager;

    /// Distribution function
    std::vector<T> hostDistributions;
    T* deviceCollision = nullptr;
    T* deviceStreaming = nullptr;

    /// swap pointer for switching streaming and collision device arrays
    T* swap = nullptr;

    /// Parameters to pass to cuda kernels
    LBMParams<T> hostParams;
    LBMParams<T>* deviceParams = nullptr;
public:
    /// Constructor
    LBMGrid(
        std::unique_ptr<LBMModel<T>>&& model,
        std::unique_ptr<CollisionModel<T>>&& collision, 
        std::unique_ptr<GridGeometry2D<T>>&& geometry, 
        std::unique_ptr<BoundaryConditionManager<T>>&& boundary
    );
    /// Destructor
    ~LBMGrid();
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

#include "lbmGrid.hh"

#endif // LATTICE_GRID_H
