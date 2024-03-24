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
    src/core/constants.h
    src/core/cell.h
    src/core/cell.hh
    src/core/gridGeometry.h
    src/core/gridGeometry.hh
    src/core/lbmGrid.h
    src/core/lbmGrid.hh
    src/core/lbmModel.h
    src/core/lbmModel.hh
    src/core/simulation.h
    src/core/simulation.hh
    src/cuda/cuda.h
    src/cuda/cuda.cu
)

# Example: LidDrivenCavity2D
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_61)  # Example for compute capability 6.1
CUDA_ADD_EXECUTABLE(LidDrivenCavity2D examples/LidDrivenCavity2D/main.cpp ${SOURCES})
target_link_libraries(LidDrivenCavity2D ${CUDA_LIBRARIES})
set_target_properties(LidDrivenCavity2D PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/LidDrivenCavity2D/build")
set_target_properties(LidDrivenCavity2D PROPERTIES CUDA_ARCHITECTURES "native")


## Example: Fluid-Heat Coupling
#add_executable(fluid_heat_coupling examples/fluid_heat_coupling/main.cpp ${FLUID_SIMULATION_SOURCES} ${FLUID_SIMULATION_HEADERS})
