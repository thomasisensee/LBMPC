cmake_minimum_required(VERSION 3.10)
project(LBMPC LANGUAGES CUDA CXX)

# Find CUDA package
find_package(CUDA REQUIRED)

# Include directories for headers
include_directories(src)

# Include directories for CUDA
include_directories(${CUDA_INCLUDE_DIRS})

# Add source files
set(SOURCES
    src/constants.h
    src/cell.h
    src/cell.hh
    src/gridGeometry.h
    src/gridGeometry.hh
    src/latticeGrid.h
    src/latticeGrid.hh
    src/lbmModel.h
    src/lbmModel.hh
    src/simulation.h
    src/simulation.hh
    src/cuda.h
    src/cuda.cu
)

# Example: LidDrivenCavity2D
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_61)  # Example for compute capability 6.1
CUDA_ADD_EXECUTABLE(LidDrivenCavity2D examples/LidDrivenCavity2D/main.cpp ${SOURCES})
target_link_libraries(LidDrivenCavity2D ${CUDA_LIBRARIES})
set_target_properties(LidDrivenCavity2D PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/LidDrivenCavity2D/build")
set_target_properties(LidDrivenCavity2D PROPERTIES CUDA_ARCHITECTURES "native")


## Example: Fluid-Heat Coupling
#add_executable(fluid_heat_coupling examples/fluid_heat_coupling/main.cpp ${FLUID_SIMULATION_SOURCES} ${FLUID_SIMULATION_HEADERS})
