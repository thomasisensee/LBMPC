#ifndef BOUNDARY_CONDITIONS_HH
#define BOUNDARY_CONDITIONS_HH

#include <cuda_runtime.h>

#include "boundaryConditions.h"
#include "core/constants.h"
#include "cuda/cudaErrorHandler.cuh"

/**********************/
/***** Base class *****/
/**********************/
template<typename T>
BoundaryCondition<T>::BoundaryCondition(BoundaryLocation loc) : _location(loc) {}

template<typename T>
BoundaryLocation BoundaryCondition<T>::getLocation() const {
    return _location;
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
        lbModel->getPopulationPtr(this->getLocation()),
        lbModel->getOppositePopualationPtr(),
        nullptr,
        this->_location
    );

    // Set block and grid size for cuda kernel execution
    this->_threadsPerBlock = THREADS_PER_BLOCK_DIMENSION*THREADS_PER_BLOCK_DIMENSION;
    if (this->_location == BoundaryLocation::EAST || this->_location == BoundaryLocation::WEST) {
        this->_numBlocks = (lbParams.Ny + this->_threadsPerBlock - 1) / this->_threadsPerBlock;
    } else {
        this->_numBlocks = (lbParams.Nx + this->_threadsPerBlock - 1) / this->_threadsPerBlock;
    }
}

template<typename T>
void BoundaryCondition<T>::printBoundaryLocation() const {
    std::cout << "== Boundary Location: " << boundaryLocationToString(_location) << "\t==" << std::endl;
}

template<typename T>
void BoundaryCondition<T>::print() const {
    printBoundaryLocation();
    std::cout << "== Condition: " << "Base" << "\t=="  << std::endl;
}


/**************************************/
/***** Derived class 01: Periodic *****/
/**************************************/
template<typename T>
PeriodicBoundary<T>::PeriodicBoundary(BoundaryLocation loc) : BoundaryCondition<T>(loc) {}

template<typename T>
void PeriodicBoundary<T>::apply(T* lbmField) {

}

template<typename T>
void PeriodicBoundary<T>::print() const {
    BoundaryCondition<T>::printBoundaryLocation();
    std::cout << "== Condition: " << "Periodic" << "\t=="  << std::endl;
}

/*****************************************/
/***** Derived class 02: Bounce Back *****/
/*****************************************/
template<typename T>
BounceBack<T>::BounceBack(BoundaryLocation loc) : BoundaryCondition<T>(loc) {}

template<typename T>
void BounceBack<T>::apply(T* lbmField) {
    //std::cout << boundaryLocationToString(this->_location) << "| V = {" << this->_params.getHostParams().WALL_VELOCITY[0] << ", " << this->_params.getHostParams().WALL_VELOCITY[1] << "}"  << std::endl;
    dim3 blockSize(this->_threadsPerBlock);
    dim3 gridSize(this->_numBlocks);
    applyBounceBackCaller(lbmField, this->_params.getDeviceParams(), gridSize, blockSize);
}

template<typename T>
void BounceBack<T>::print() const {
    BoundaryCondition<T>::printBoundaryLocation();
    std::cout << "== Condition: " << "Bounce-back" << "\t=="  << std::endl;
}

/**********************************************************/
/***** Derived class 03: Fixed Velocity (Bounce Back) *****/
/**********************************************************/
template<typename T>
MovingWall<T>::MovingWall(BoundaryLocation loc, const std::vector<T>& velocity) : BounceBack<T>(loc), _wallVelocity(velocity) {}

template<typename T>
void MovingWall<T>::prepareKernelParams(const LBParams<T>& lbmParams, const LBModel<T>* lbModel) {
    BoundaryCondition<T>::prepareKernelParams(lbmParams, lbModel);
    this->_params.setWallVelocity(getWallVelocity());
}

template<typename T>
const std::vector<T>& MovingWall<T>::getWallVelocity() const {
    return _wallVelocity;
}

template<typename T>
void MovingWall<T>::print() const {
    BoundaryCondition<T>::printBoundaryLocation();
    std::cout << "== Condition: " << "Bounce-Back with fixed velocity" << "\t=="  << std::endl;
    std::cout << "== Velocity = {" << getWallVelocity()[0] << ", " << getWallVelocity()[1] << "}\t=="  << std::endl;
}
/*
template<typename T>
void MovingWall<T>::apply(T* lbmField) {
    std::cout << boundaryLocationToString(this->_location) << "| V = {" << this->_params.getHostParams().WALL_VELOCITY[0] << "}"  << std::endl;
    dim3 blockSize(this->_threadsPerBlock);
    dim3 gridSize(this->_numBlocks);
    applyBounceBackCaller(lbmField, this->_params.getDeviceParams(), gridSize, blockSize);
}*/


/*************************/
/***** Wrapper class *****/
/*************************/
template<typename T>
BoundaryConditionManager<T>::BoundaryConditionManager() {

}

template<typename T>
void BoundaryConditionManager<T>::addBoundaryCondition(std::unique_ptr<BoundaryCondition<T>> condition) {
    for (const auto& existingCondition : boundaryConditions) {
        if (condition->getLocation() == existingCondition->getLocation()) {
            std::cerr << "Error: Boundary condition already exists for location " << boundaryLocationToString(condition->getLocation()) << std::endl;
            exit(EXIT_FAILURE);
        }
    }   

    boundaryConditions.push_back(std::move(condition));
}

template<typename T>
void BoundaryConditionManager<T>::prepareKernelParams(const LBParams<T>& lbParams, const LBModel<T>* lbModel) {
    for (const auto& condition : boundaryConditions) {
        condition->prepareKernelParams(lbParams, lbModel);
    }
}

template<typename T>
void BoundaryConditionManager<T>::apply(T* lbmField) {
    for (const auto& condition : boundaryConditions) {
        condition->apply(lbmField);
    }
}

template<typename T>
void BoundaryConditionManager<T>::print() const {
    std::cout << "====== Boundary conditions =======" << std::endl;
    for (const auto& condition : boundaryConditions) {
        condition->print();
    }
    std::cout << "==================================\n" << std::endl;
}

#endif // BOUNDARY_CONDITIONS_HH