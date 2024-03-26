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

auto collisionModel = std::make_unique<MRTCHMCollisionModel<T>>();
collisionModel->print();

// ===============================
// === Prepare domain geometry ===
// ===============================
auto gridGeometry = std::make_unique<GridGeometry2D<T>>(0.0,0.0,0.00793650793,126,126);
gridGeometry->print();

// ===================================
// === Prepare boundary conditions ===
// ===================================
auto boundaryConditionManager = std::make_unique<BoundaryConditionManager<T>>();
std::vector<T> wallVelocity = {1.0,0.0};
boundaryConditionManager->addBoundaryCondition(BoundaryType::West, "bounceBack", std::make_unique<BounceBack<T>>());
boundaryConditionManager->addBoundaryCondition(BoundaryType::East, "bounceBack", std::make_unique<BounceBack<T>>());
boundaryConditionManager->addBoundaryCondition(BoundaryType::South, "bounceBack", std::make_unique<BounceBack<T>>());
boundaryConditionManager->addBoundaryCondition(BoundaryType::North, "fixedVelocity", std::make_unique<FixedVelocityBoundary<T>>(wallVelocity));
boundaryConditionManager->print();


// ========================
// === Setup simulation ===
// ========================
auto lbmGrid = std::make_unique<LBMGrid<T>>(
    std::move(lbmModel),
    std::move(collisionModel), 
    std::move(gridGeometry), 
    std::move(boundaryConditionManager)
);


return 0;
}
