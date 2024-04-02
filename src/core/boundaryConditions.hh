#ifndef BOUNDARY_CONDITIONS_HH
#define BOUNDARY_CONDITIONS_HH

#include <cuda_runtime.h>

#include "boundaryConditions.h"
#include "cuda/cudaErrorHandler.cuh"

/**********************/
/***** Base class *****/
/**********************/
template<typename T>
BoundaryCondition<T>::BoundaryCondition(BoundaryLocation loc) : location(loc) {}

template<typename T>
BoundaryCondition<T>::~BoundaryCondition() {}

template<typename T>
BoundaryLocation BoundaryCondition<T>::getLocation() const {
    return location;
}

template<typename T>
void BoundaryCondition<T>::prepareKernelParams(const LBParams<T>& lbParams, const  LBModel<T>* lbModel) {
    // Set kernel parameters (and duplicate on device)
    _params.setValues(
        lbParams.D,
        lbParams.Nx,
        lbParams.Ny,
        lbParams.Q,
        lbParams.LATTICE_VELOCITIES,
        lbParams.LATTICE_WEIGHTS,
        lbModel->OPPOSITE_POPULATION,
        nullptr,
        this->location
    );

    // Set block and grid size for cuda kernel execution
    this->threadsPerBlock = THREADS_PER_BLOCK_DIMENSION*THREADS_PER_BLOCK_DIMENSION;
    if (this->location == BoundaryLocation::EAST || this->location == BoundaryLocation::WEST) {
        this->numBlocks = (lbParams.Ny + this->threadsPerBlock - 1) / this->threadsPerBlock;
    } else {
        this->numBlocks = (lbParams.Nx + this->threadsPerBlock - 1) / this->threadsPerBlock;
    }
}

/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
PeriodicBoundary<T>::PeriodicBoundary(BoundaryLocation loc) : BoundaryCondition<T>(loc) {}

template<typename T>
void PeriodicBoundary<T>::apply(T* lbmField) {

}

template<typename T>
BounceBack<T>::BounceBack(BoundaryLocation loc) : BoundaryCondition<T>(loc) {}

template<typename T>
void BounceBack<T>::apply(T* lbmField) {
    dim3 blockSize(this->threadsPerBlock);
    dim3 gridSize(this->numBlocks);
    applyBounceBackCaller(lbmField, this->_params.getDeviceParams(), gridSize, blockSize);
}

template<typename T>
FixedVelocityBoundary<T>::FixedVelocityBoundary(BoundaryLocation loc, const std::vector<T>& velocity) : BounceBack<T>(loc), wallVelocity(velocity) {}

template<typename T>
void FixedVelocityBoundary<T>::prepareKernelParams(const LBParams<T>& lbmParams, const LBModel<T>* lbModel) {
    BoundaryCondition<T>::prepareKernelParams(lbmParams, lbModel);
    this->_params.setWallVelocity(this->wallVelocity.data()); // Assign the address of the first element

}


/*************************/
/***** Wrapper class *****/
/*************************/
template<typename T>
BoundaryConditionManager<T>::BoundaryConditionManager() {

}

template<typename T>
void BoundaryConditionManager<T>::addBoundaryCondition(const std::string& name, std::unique_ptr<BoundaryCondition<T>> condition) {
    BoundaryLocation loc = condition->getLocation();
    boundaryConditions[loc][name] = std::move(condition);
}

template<typename T>
void BoundaryConditionManager<T>::prepareKernelParams(const LBParams<T>& lbParams, const LBModel<T>* lbModel) {
    // Iterate over each boundary location
    for (auto& boundaryConditionsPair : boundaryConditions) {
        // Now iterate over each condition for this boundary
        for (auto& conditionPair : boundaryConditionsPair.second) {
            // Apply the boundary condition
            conditionPair.second->prepareKernelParams(lbParams, lbModel);
        }
    }
}

template<typename T>
void BoundaryConditionManager<T>::apply(T* lbmField) {
    // Iterate over each boundary location
    for (auto& boundaryConditionsPair : boundaryConditions) {
        // Now iterate over each condition for this boundary
        for (auto& conditionPair : boundaryConditionsPair.second) {
            // Apply the boundary condition
            conditionPair.second->apply(lbmField);
        }
    }
}

template<typename T>
void BoundaryConditionManager<T>::print() const {
    std::cout << "====== Boundary conditions =======" << std::endl;
    for (const auto& boundaryConditionsPair : boundaryConditions) {
        std::cout << "== Boundary Location: " << boundaryLocationToString(boundaryConditionsPair.first) << "\t==" << std::endl;
        for (const auto& conditionPair : boundaryConditionsPair.second) {
            std::cout << "== Condition: " << conditionPair.first << "\t=="  << std::endl;
            std::cout << "==\t\t\t\t==\n";
        }
    }
    std::cout << "==================================\n" << std::endl;
}
#endif // BOUNDARY_CONDITIONS_HH
