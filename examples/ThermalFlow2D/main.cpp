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
    using MOMENTUM_DESCRIPTOR = descriptors::ScalarD2Q9<T>;
    using THERMAL_DESCRIPTOR = descriptors::ScalarD2Q5<T>;

    // =========================
    // === Set grid geometry ===
    // =========================
    unsigned int nx = 126, ny = 126;
    T dx = 1.0 / nx;
    auto gridGeometry = std::make_shared<GridGeometry2D<T>>(dx, nx, ny);
    //gridGeometry->printParameters();

    // =============================================================================
    // === Determine parameters: time step, omega etc. for momentum conservation ===
    // =============================================================================
    T prandtlNumber = 0.71;
    T tauShear = 0.7;
    T omegaShear = 1.0 / tauShear;
    T reynoldsNumber = 1.0 / prandtlNumber;
	T dt = (tauShear - 0.5) * dx * dx * reynoldsNumber * descriptors::cs2<T,MOMENTUM_DESCRIPTOR::LATTICE::D,MOMENTUM_DESCRIPTOR::LATTICE::Q>();


    // ============================================================================
    // === Determine parameters: time step, omega etc. for thermal distribution ===
    // ============================================================================
    T tauThermal = (tauShear - 0.5) / prandtlNumber + 0.5;
    T omegaThermal = 1.0 / tauThermal;
    T initialTemperature = 0.0;
    T wallTemperatureLeft = 1.0;
    T wallTemperatureRight = 0.0;

    // ======================================
    // === Set other necessary components ===
    // ======================================
    auto collisionModelMomentum = std::make_unique<CollisionBGK<T,MOMENTUM_DESCRIPTOR>>(omegaShear);
    auto collisionModelThermal  = std::make_unique<CollisionBGK<T,THERMAL_DESCRIPTOR>>(omegaThermal);

    //collisionModel->printParameters();

    auto boundaryConditionManagerMomentum = std::make_unique<BoundaryConditionManager<T,MOMENTUM_DESCRIPTOR>>();
    boundaryConditionManagerMomentum->addBoundaryCondition(std::make_unique<BounceBack<T,MOMENTUM_DESCRIPTOR>>(BoundaryLocation::WEST));
    boundaryConditionManagerMomentum->addBoundaryCondition(std::make_unique<BounceBack<T,MOMENTUM_DESCRIPTOR>>(BoundaryLocation::EAST));
    boundaryConditionManagerMomentum->addBoundaryCondition(std::make_unique<BounceBack<T,MOMENTUM_DESCRIPTOR>>(BoundaryLocation::SOUTH));
    boundaryConditionManagerMomentum->addBoundaryCondition(std::make_unique<BounceBack<T,MOMENTUM_DESCRIPTOR>>(BoundaryLocation::NORTH));
    //boundaryConditionManagerMomentum->printParameters();

    auto boundaryConditionManagerThermal = std::make_unique<BoundaryConditionManager<T,THERMAL_DESCRIPTOR>>();
    boundaryConditionManagerThermal->addBoundaryCondition(std::make_unique<AntiBounceBack<T,THERMAL_DESCRIPTOR>>(BoundaryLocation::WEST, wallTemperatureLeft));
    boundaryConditionManagerThermal->addBoundaryCondition(std::make_unique<AntiBounceBack<T,THERMAL_DESCRIPTOR>>(BoundaryLocation::EAST, wallTemperatureRight));
    boundaryConditionManagerThermal->addBoundaryCondition(std::make_unique<BounceBack<T,THERMAL_DESCRIPTOR>>(BoundaryLocation::SOUTH));
    boundaryConditionManagerThermal->addBoundaryCondition(std::make_unique<BounceBack<T,THERMAL_DESCRIPTOR>>(BoundaryLocation::NORTH));
    //boundaryConditionManagerThermal->printParameters();

 
    auto lbGridMomentum = std::make_unique<LBGrid<T,MOMENTUM_DESCRIPTOR>>(gridGeometry, std::move(collisionModelMomentum), std::move(boundaryConditionManagerMomentum));
    auto lbGridThermal = std::make_unique<LBGrid<T,THERMAL_DESCRIPTOR>>(gridGeometry, std::move(collisionModelThermal), std::move(boundaryConditionManagerThermal), initialTemperature);
    //lbGridMomentum->printParameters();
    //lbGridThermal->printParameters();


    // ======================
    // === Run simulation ===
    // ======================
    std::string outputDirectory = "./output";
    std::string baseFileName = "thermalFlow";
    T simTime = 1.;
    unsigned int nOut = 10;
    auto vtkWriter = std::make_unique<VTKWriter>(outputDirectory, baseFileName);
    auto simulation = std::make_unique<LBCoupledSimulation<T,MOMENTUM_DESCRIPTOR,THERMAL_DESCRIPTOR>>(std::move(lbGridMomentum), std::move(lbGridThermal), std::move(vtkWriter), dt, simTime, nOut);
    simulation->printParameters();

    simulation->run();

    return EXIT_SUCCESS;
}