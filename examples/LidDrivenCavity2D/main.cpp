#include <stdio.h>
#include <memory> // For std::unique_ptr and std::make_unique

#include "core/constants.h"
#include "core/lb/lbConstants.h"
#include "core/gridGeometry.h"
#include "core/simulation.h"
#include "core/lb/lbModel.h"
#include "core/lb/collisionModel.h"
#include "core/lb/boundaryConditions.h"
#include "core/lb/lbGrid.h"
#include "cuda/cudaUtilities.cuh"

#include "io/configurationManager.h"

using FLOATING_POINT_TYPE = float;
using T = FLOATING_POINT_TYPE;

int main() {
    // =======================
    // === Set CUDA device ===
    // =======================
    SetDevice();

    // ============================================
    // === Read input file and build simulation ===
    // ============================================
    ConfigurationManager configManager("/home/thomas/Work/code/ff2D/LBMPC/examples/LidDrivenCavity2D/config.xml");
    auto simulation = configManager.buildSimulation<T>();

    simulation->printParameters();

    // ======================
    // === Run simulation ===
    // ======================
    simulation->run();

    return EXIT_SUCCESS;
}