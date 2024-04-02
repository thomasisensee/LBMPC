#include <stdio.h>
#include <memory> // For std::unique_ptr and std::make_unique

#include "core/lbModel.h"
#include "core/collisionModel.h"
#include "core/gridGeometry.h"
#include "core/boundaryConditions.h"
#include "core/lbGrid.h"
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
auto lbModel = std::make_unique<D2Q9<T>>(); // Create instances of the components. Since we're passing these to LBMGrid which takes ownership, we use std::make_unique to create unique_ptr instances.
lbModel->print();

T omegaShear = 0.7;
T omegaBulk = omegaShear;
auto collisionModel = std::make_unique<CollisionBGK<T>>(omegaShear);
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
boundaryConditionManager->addBoundaryCondition("bounceBack", std::make_unique<BounceBack<T>>(BoundaryLocation::WEST));
boundaryConditionManager->addBoundaryCondition("bounceBack", std::make_unique<BounceBack<T>>(BoundaryLocation::EAST));
boundaryConditionManager->addBoundaryCondition("bounceBack", std::make_unique<BounceBack<T>>(BoundaryLocation::SOUTH));
boundaryConditionManager->addBoundaryCondition("fixedVelocity", std::make_unique<FixedVelocityBoundary<T>>(BoundaryLocation::NORTH, wallVelocity));
boundaryConditionManager->print();

// ======================
// === Setup LBM Grid ===
// ======================
auto lbGrid = std::make_unique<LBGrid<T>>(
    std::move(lbModel),
    std::move(collisionModel), 
    std::move(gridGeometry), 
    std::move(boundaryConditionManager)
);

// ========================
// === Setup simulation ===
// ========================
LBFluidSimulation simulation = LBFluidSimulation<T>(std::move(lbGrid));

simulation.run();


return 0;
}
