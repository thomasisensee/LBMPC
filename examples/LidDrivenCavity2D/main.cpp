#include <stdio.h>
#include <memory>
#include <cuda_runtime.h>

#include "core/lb/lbConstants.h"
#include "core/descriptors/descriptors.h"
#include "core/gridGeometry.h"
#include "core/lb/lbModel.h"
#include "core/lb/collisionModel.h"
#include "core/lb/boundaryConditions.h"
#include "cuda/cudaUtilities.h"

//#include "io/configurationManager.h"

using FLOATING_POINT_TYPE = float;
using T = FLOATING_POINT_TYPE;

int main() {
    // =======================
    // === Set CUDA device ===
    // =======================
    SetDevice();

    // ================================
    // === Set necessary components ===
    // ================================
    unsigned int nx = 126, ny = 126;
    T dx = 1.0 / nx;
    auto gridGeometry = std::make_unique<GridGeometry2D<T>>(dx, nx, ny);
    gridGeometry->printParameters();

    auto lbModel = std::make_unique<LBModel<T, D2Q9Descriptor<T>>>();
    lbModel->printParameters();

    T tauShear = 0.7;
    T tauBulk = tauShear;
    T omegaShear = 1.0 / tauShear;
    T omegaBulk = 1.0 / tauBulk;
    auto collisionModel = std::make_unique<CollisionBGK<T>>(omegaShear);
    collisionModel->printParameters();

    T reynoldsNumber = 100.0;
	T dt = (tauShear - 0.5) * dx * dx * reynoldsNumber * C_S_POW2;
    std::vector<T> wallVelocity = {dt / dx, 0.0};
    auto boundaryConditionManager = std::make_unique<BoundaryConditionManager<T>>();
    boundaryConditionManager->addBoundaryCondition(std::make_unique<BounceBack<T>>(BoundaryLocation::WEST));
    boundaryConditionManager->addBoundaryCondition(std::make_unique<BounceBack<T>>(BoundaryLocation::EAST));
    boundaryConditionManager->addBoundaryCondition(std::make_unique<BounceBack<T>>(BoundaryLocation::SOUTH));
    boundaryConditionManager->addBoundaryCondition(std::make_unique<MovingWall<T>>(BoundaryLocation::NORTH, wallVelocity));
    boundaryConditionManager->printParameters();

    // ======================
    // === Run simulation ===
    // ======================
    

    return EXIT_SUCCESS;
}