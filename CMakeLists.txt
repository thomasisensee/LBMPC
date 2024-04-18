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
    src/core/platform_definitions.h
    src/core/descriptors/descriptors.h
    src/core/descriptors/latticeDescriptors.h
    src/core/descriptors/fieldTags.h
    src/core/functors/functors.h
    src/core/functors/functors.hh
    src/core/utilities.h
    src/core/utilities.cpp
    src/core/gridGeometry.h
    src/core/gridGeometry.hh
    src/core/lb/lbGrid.h
    src/core/lb/lbGrid.hh
    src/core/lb/collisionModel.h
    src/core/lb/collisionModel.hh
    src/core/boundary/boundaryConditions.h
    src/core/boundary/boundaryConditions.hh
    src/core/simulation.h
    src/core/simulation.hh
    src/core/kernelParameters.h
    src/core/kernelParameters.hh
    src/io/vtkWriter.h
    src/io/vtkWriter.hh
    #src/io/configurationManager.h
    #src/io/configurationManager.hh
    src/cuda/cudaErrorHandler.cuh
    src/cuda/cell.h
    src/cuda/cell.hh
    src/cuda/cudaKernels.cuh
    src/cuda/cudaKernels.cu
    src/cuda/cudaUtilities.h
    src/cuda/cudaUtilities.cpp
    src/utilities/fraction.h
)

# Define the tinyxml2 library
add_library(tinyxml2 STATIC external/tinyxml2/tinyxml2.cpp)
target_include_directories(tinyxml2 PUBLIC external/tinyxml2)

# Example: LidDrivenCavity2D
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_61)  # Example for compute capability 6.1
CUDA_ADD_EXECUTABLE(LidDrivenCavity2D examples/LidDrivenCavity2D/main.cpp ${SOURCES})
target_link_libraries(LidDrivenCavity2D ${CUDA_LIBRARIES} tinyxml2)
set_target_properties(LidDrivenCavity2D PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/LidDrivenCavity2D/build")
set_target_properties(LidDrivenCavity2D PROPERTIES CUDA_ARCHITECTURES "native")

# Example: ThermalDiffusion2D
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_61)  # Example for compute capability 6.1
#CUDA_ADD_EXECUTABLE(ThermalDiffusion2D examples/ThermalDiffusion2D/main.cpp ${SOURCES})
#target_link_libraries(ThermalDiffusion2D ${CUDA_LIBRARIES} tinyxml2)
#set_target_properties(ThermalDiffusion2D PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/ThermalDiffusion2D/build")
#set_target_properties(ThermalDiffusion2D PROPERTIES CUDA_ARCHITECTURES "native")