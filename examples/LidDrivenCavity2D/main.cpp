#include <stdio.h>
#include <memory> // For std::unique_ptr and std::make_unique

#include "core/lbmModel.h"
#include "core/collisionModel.h"
#include "core/gridGeometry.h"
#include "core/boundaryConditions.h"
#include "core/lbmGrid.h"
#include "core/simulation.h"
#include "cuda/cudaUtilities.cuh"

using FLOATING_POINT_TYPE = float;
using T = FLOATING_POINT_TYPE;

int main( int argc, char* argv[] )
{

// =======================
// === Set CUDA device ===
// =======================
SetDevice();

// ============================================
// === Define LBM model and collision model ===
// ============================================
auto lbmModel = std::make_unique<D2Q9<T>>(); // Create instances of the components. Since we're passing these to LBMGrid which takes ownership, we use std::make_unique to create unique_ptr instances.
lbmModel->print();

T omegaShear = 0.7;
T omegaBulk = omegaShear;
auto collisionModel = std::make_unique<CollisionCHM<T>>(omegaShear,omegaBulk);
collisionModel->print();

// ===============================
// === Prepare domain geometry ===
// ===============================
auto gridGeometry = std::make_unique<GridGeometry2D<T>>(0.00793650793,126,126);
gridGeometry->print();

// ===================================
// === Prepare boundary conditions ===
// ===================================
auto boundaryConditionManager = std::make_unique<BoundaryConditionManager<T>>();
std::vector<T> wallVelocity = {1.0,0.0};
boundaryConditionManager->addBoundaryCondition(BoundaryLocation::WEST, "bounceBack", std::make_unique<BounceBack<T>>());
boundaryConditionManager->addBoundaryCondition(BoundaryLocation::EAST, "bounceBack", std::make_unique<BounceBack<T>>());
boundaryConditionManager->addBoundaryCondition(BoundaryLocation::SOUTH, "bounceBack", std::make_unique<BounceBack<T>>());
boundaryConditionManager->addBoundaryCondition(BoundaryLocation::NORTH, "fixedVelocity", std::make_unique<FixedVelocityBoundary<T>>(wallVelocity));
boundaryConditionManager->print();

// ======================
// === Setup LBM Grid ===
// ======================
auto lbmGrid = std::make_unique<LBMGrid<T>>(
    std::move(lbmModel),
    std::move(collisionModel), 
    std::move(gridGeometry), 
    std::move(boundaryConditionManager)
);

// ========================
// === Setup simulation ===
// ========================
LBMFluidSimulation simulation = LBMFluidSimulation<T>(std::move(lbmGrid));

simulation.run();


return 0;
}
