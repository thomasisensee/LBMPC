#include <stdio.h>

#include "cuda/cudaUtilities.h"

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