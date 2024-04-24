#include <stdio.h>
#include <memory>
#include <cuda_runtime.h>

#include "core/descriptors/descriptors.h"
#include "core/descriptors/aliases.h"
#include "core/grid/gridGeometry2D.h"
#include "core/lb/collisionModel.h"
#include "core/boundary/boundaryConditions.h"
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
    //using DESCRIPTOR = descriptors::ScalarD2Q9<T>;
    using DESCRIPTOR = descriptors::ScalarD2Q5<T>;

    // =========================
    // === Set grid geometry ===
    // =========================
    unsigned int nx = 126, ny = 126;
    T dx = 1.0 / nx;
    auto gridGeometry = std::make_shared<GridGeometry2D<T>>(dx, nx, ny);
    //gridGeometry->printParameters();

    // ===================================================
    // === Determine parameters: time step, omega etc. ===
    // ===================================================
    T tauShear = 0.7;
    T omegaShear = 1.0 / tauShear;
    T pecletNumber = 1.0;
	T dt = (tauShear - 0.5) * dx * dx * pecletNumber * descriptors::cs2<T,DESCRIPTOR::LATTICE::D,DESCRIPTOR::LATTICE::Q>();
    T initialTemperature = 0.0;
    T wallTemperatureLeft = 1.0;
    T wallTemperatureRight = 0.0;

    // ======================================
    // === Set other necessary components ===
    // ======================================
    auto collisionModel = std::make_unique<CollisionBGK<T,DESCRIPTOR>>(omegaShear);
    //collisionModel->printParameters();

    auto boundaryConditionManager = std::make_unique<BoundaryConditionManager<T,DESCRIPTOR>>();
    boundaryConditionManager->addBoundaryCondition(std::make_unique<AntiBounceBack<T,DESCRIPTOR>>(BoundaryLocation::WEST, wallTemperatureLeft));
    boundaryConditionManager->addBoundaryCondition(std::make_unique<AntiBounceBack<T,DESCRIPTOR>>(BoundaryLocation::EAST, wallTemperatureRight));
    boundaryConditionManager->addBoundaryCondition(std::make_unique<BounceBack<T,DESCRIPTOR>>(BoundaryLocation::SOUTH));
    boundaryConditionManager->addBoundaryCondition(std::make_unique<BounceBack<T,DESCRIPTOR>>(BoundaryLocation::NORTH));
    //boundaryConditionManager->printParameters();

 
    auto lbGrid = std::make_unique<LBGrid<T,DESCRIPTOR>>(gridGeometry, std::move(collisionModel), std::move(boundaryConditionManager), initialTemperature);
    //lbGrid->printParameters();


    // ======================
    // === Run simulation ===
    // ======================
    std::string outputDirectory = "./output";
    std::string baseFileName = "thermalDiffusion";
    T simTime = 1.;
    unsigned int nOut = 10;
    auto vtkWriter = std::make_unique<VTKWriter>(outputDirectory, baseFileName);
    auto simulation = std::make_unique<LBFluidSimulation<T,DESCRIPTOR>>(std::move(lbGrid), std::move(vtkWriter), dt, simTime, nOut);
    simulation->printParameters();

    simulation->run();

    return EXIT_SUCCESS;
}