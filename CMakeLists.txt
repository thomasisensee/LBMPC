cmake_minimum_required(VERSION 3.10)
project(LBMPC)

# Add source files for fluid simulation module
file(GLOB_RECURSE LBMPC_SOURCES src/*.hh)
file(GLOB_RECURSE LBMPC_HEADERS src/*.h)

# Include directories for headers
include_directories(src)

# Example: LidDrivenCavity2D
add_executable(LidDrivenCavity2D examples/LidDrivenCavity2D/main.cpp ${LBMPC_SOURCES} ${LBMPC_HEADERS})
set_target_properties(LidDrivenCavity2D PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/LidDrivenCavity2D/build")

## Example: Fluid-Heat Coupling
#add_executable(fluid_heat_coupling examples/fluid_heat_coupling/main.cpp ${FLUID_SIMULATION_SOURCES} ${FLUID_SIMULATION_HEADERS})

