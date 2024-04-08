#ifndef LATTICE_GRID_H
#define LATTICE_GRID_H

#include <stdio.h>
#include <vector>

#include "lbModel.h"
#include "collisionModel.h"
#include "boundaryConditions.h"
#include "core/gridGeometry.h"
#include "core/kernelParameters.h"

template<typename T>
class LBGrid {
private:
    /// Required model objects
    std::unique_ptr<LBModel<T>> _lbModel;
    std::unique_ptr<CollisionModel<T>> _collisionModel;
    std::unique_ptr<GridGeometry2D<T>> _gridGeometry;
    std::unique_ptr<BoundaryConditionManager<T>> _boundaryConditionManager;

    /// Distribution function, both for host and device
    std::vector<T> _hostDistributions;
    T* _deviceCollision = nullptr;
    T* _deviceStreaming = nullptr;

    /// Fields for zeroth and first moments, both for host and device
    std::vector<T> _hostZerothMoment;
    std::vector<T> _hostFirstMoment;
    T* _deviceZerothMoment = nullptr;
    T* _deviceFirstMoment = nullptr;

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

    /// Getter for _hostDistributions
    std::vector<T>& getHostDistributions();

    /// Getter for host-side moment vectors
    std::vector<T>& getHostZerothMoment();
    std::vector<T>& getHostFirstMoment();

    /// Getter for _deviceCollision and _deviceStreaming
    T* getDeviceCollision() const;
    T* getDeviceStreaming() const;

    /// Getter for device-side moment pointers
    T* getDeviceZerothMoment() const;
    T* getDeviceFirstMoment() const;

    /// Getter for gridGeometry
    const GridGeometry2D<T>& getGridGeometry() const;

    /// Memory operations
    void allocateHostData();
    void allocateDeviceData();
    void cleanupDevice();
    void prepareKernelParams();
    void fetchZerothMoment();
    void fetchFirstMoment();

    /// Host-Device communication
    void copyToDevice();
    void copyToHost();

    /// Main computations
    void initializeDistributions();
    void performCollisionStep();
    void performStreamingStep();
    void applyBoundaryConditions();
    void computeMoments();
    void computeZerothMoment();
    void computeFirstMoment();
    static unsigned int pos(unsigned int i, unsigned int j, unsigned int Nx);
};

#include "lbGrid.hh"

#endif // LB_GRID_H