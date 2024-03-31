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
BoundaryCondition<T>::~BoundaryCondition() {
    if (deviceParams != nullptr) {
        cudaErrorCheck(cudaFree(deviceParams));
    }
}

template<typename T>
BoundaryLocation BoundaryCondition<T>::getLocation() const {
    return location;
}

template<typename T>
void BoundaryCondition<T>::prepareKernelParams(LBParams<T>* lbmParams) {
    this->hostParams.D                  = lbmParams->D;
    this->hostParams.Nx                 = lbmParams->Nx;
    this->hostParams.Ny                 = lbmParams->Ny;
    this->hostParams.Q                  = lbmParams->Q;
    this->hostParams.LATTICE_VELOCITIES = lbmParams->LATTICE_VELOCITIES;
    this->hostParams.LATTICE_WEIGHTS    = lbmParams->LATTICE_WEIGHTS;
    this->hostParams.wallVelocity       = nullptr;
    this->hostParams.location           = this->location;
}

template<typename T>
void BoundaryCondition<T>::copyKernelParamsToDevice() {
    // Allocate device memory for lattice velocities and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = this->hostParams.Q * this->hostParams.D * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = this->hostParams.Q * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));

    // Prepare the host-side copy of LBParams with device pointers
    BoundaryParams<T> paramsTemp = hostParams; // Use a temporary host copy
    paramsTemp.LATTICE_VELOCITIES = deviceLatticeVelocities;
    paramsTemp.LATTICE_WEIGHTS = deviceLatticeWeights;

    // Allocate memory for the LBParams struct on the device if not already allocated
    if (deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&deviceParams, sizeof(BoundaryParams<T>)));
    }

    // Copy the prepared LBParams (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(deviceParams, &paramsTemp, sizeof(BoundaryParams<T>), cudaMemcpyHostToDevice));
}

/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
BounceBack<T>::BounceBack(BoundaryLocation loc) : BoundaryCondition<T>(loc) {}

template<typename T>
void BounceBack<T>::apply(T* lbmField) {
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIMENSION*THREADS_PER_BLOCK_DIMENSION);

    if(this->location == BoundaryLocation::EAST || this->location == BoundaryLocation::WEST) {
        dim3 numBlocks = (this->hostParams.Ny + threadsPerBlock.x - 1) / threadsPerBlock.x;
    } else {
        dim3 numBlocks = (this->hostParams.Nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    }
    
    
}

template<typename T>
FixedVelocityBoundary<T>::FixedVelocityBoundary(BoundaryLocation loc, const std::vector<T>& velocity) : BoundaryCondition<T>(loc), wallVelocity(velocity) {}

template<typename T>
void FixedVelocityBoundary<T>::prepareKernelParams(LBParams<T>* lbmParams) {
    BoundaryCondition<T>::prepareKernelParams(lbmParams);
    this->hostParams.wallVelocity = this->wallVelocity.data(); // Assign the address of the first element
}

template<typename T>
void FixedVelocityBoundary<T>::apply(T* lbmField) {

}

template<typename T>
PeriodicBoundary<T>::PeriodicBoundary(BoundaryLocation loc) : BoundaryCondition<T>(loc) {}

template<typename T>
void PeriodicBoundary<T>::apply(T* lbmField) {

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
