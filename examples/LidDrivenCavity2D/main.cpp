#include <stdio.h>
#include <memory>
#include <cuda_runtime.h>

#include "core/descriptors/latticeDescriptors.h"
#include "core/gridGeometry.h"
#include "core/lb/collisionModel.h"
#include "core/lb/boundaryConditions.h"
#include "core/lb/lbGrid.h"
#include "core/simulation.h"
#include "cuda/cudaUtilities.h"

//#include "io/configurationManager.h"

using FLOATING_POINT_TYPE = float;
using T = FLOATING_POINT_TYPE;

int main() {
    // =======================
    // === Set CUDA device ===
    // =======================
    SetDevice();

    // ==============================
    // === Set lattice descriptor ===
    // ==============================
    using D2Q9 = latticeDescriptors::D2Q9;
    const unsigned int D = 2;
    const unsigned int Q = 9;

    // ================================
    // === Set necessary components ===
    // ================================
    unsigned int nx = 126, ny = 126;
    T dx = 1.0 / nx;
    auto gridGeometry = std::make_unique<GridGeometry2D<T>>(dx, nx, ny);
    //gridGeometry->printParameters();


    T tauShear = 0.7;
    T tauBulk = tauShear;
    T omegaShear = 1.0 / tauShear;
    T omegaBulk = 1.0 / tauBulk;
    //auto collisionModel = std::make_unique<CollisionBGK<T,D,Q>>(omegaShear);
    auto collisionModel = std::make_unique<CollisionCHM<T,D,Q>>(omegaShear, omegaBulk);
    //collisionModel->printParameters();


    T reynoldsNumber = 100.0;
	T dt = (tauShear - 0.5) * dx * dx * reynoldsNumber * latticeDescriptors::cs2<T,D,Q>();
    T u = 1.0;
    T v = 0.0;
    u *= dt / dx;
    v *= dt / dx;
    std::vector<T> wallVelocity = {u, v};
    auto boundaryConditionManager = std::make_unique<BoundaryConditionManager<T,D,Q>>();
    boundaryConditionManager->setDxdt(dx/dt);
    boundaryConditionManager->addBoundaryCondition(std::make_unique<BounceBack<T,D,Q>>(BoundaryLocation::WEST));
    boundaryConditionManager->addBoundaryCondition(std::make_unique<BounceBack<T,D,Q>>(BoundaryLocation::EAST));
    boundaryConditionManager->addBoundaryCondition(std::make_unique<BounceBack<T,D,Q>>(BoundaryLocation::SOUTH));
    boundaryConditionManager->addBoundaryCondition(std::make_unique<MovingWall<T,D,Q>>(BoundaryLocation::NORTH, wallVelocity));
    //boundaryConditionManager->printParameters();

 
    auto lbGrid = std::make_unique<LBGrid<T,D,Q>>(std::move(gridGeometry), std::move(collisionModel), std::move(boundaryConditionManager));
    //lbGrid->printParameters();


    // ======================
    // === Run simulation ===
    // ======================
    std::string outputDirectory = "./output";
    std::string baseFileName = "cavity";
    T simTime = 10.0;
    unsigned int nOut = 10;
    auto vtkWriter = std::make_unique<VTKWriter>(outputDirectory, baseFileName);
    auto simulation = std::make_unique<LBFluidSimulation<T,D,Q>>(std::move(lbGrid), std::move(vtkWriter), dt, simTime, nOut);
    simulation->printParameters();

    simulation->run();

    return EXIT_SUCCESS;
}